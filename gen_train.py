import sys
import re
import json
from functools import reduce
import logging
import argparse
import subprocess

import pandas as pd
import numpy as np
from util.debug import dprint
from Bio import SeqIO

logging.basicConfig(level=logging.INFO)


####################################################################################################
####################################################################################################
def collapse_to_count(l: list) -> dict:
    '''Given list, return dict with elements as keys and counts as values'''
    keys = list(set(l))
    return {key: l.count(key) for key in keys}

####################################################################################################
####################################################################################################
FIXED_RANKS = ['domain', 'phylum', 'class', 'order', 'family', 'genus']
ROOT_PATH = 'Root'
LOWEST_RANK = 'genus'


####################################################################################################
####################################################################################################
class Node:

    # path -> Node
    # Usually something like ([A-Z][a-z]+;){1,20} -> <Node>
    # For root, it's 'Root' -> <Node>
    instances = {}

    #
    taxid_l = [] # redundant check for uniqueness

    

####################################################################################################
    def __init__(self, path, taxid, rank, rankName_dup=None):
        '''
        Root comes in as <'Root', 0, 'rootrank'>
        Everything else comes in from tax_slv_ssu_138.txt as <'(Taxname;)+', \d+, 'rank'>
        '''
        #
        self.path = path # with exception of 'Root', begins with domain and ends in ';'
        self.taxid = int(taxid) # from SILVA, mostly stable across releases
        self.rank = rank

        #
        self.instances[path] = self

        # rank x name
        self.rankName_dup = rankName_dup

        # leaf
        self.is_leaf = True
        if self.path != ROOT_PATH:
            self.parent_node.is_leaf = False

        # taxids
        assert taxid not in self.taxid_l
        self.taxid_l.append(taxid)

####################################################################################################
    @property
    def depth(self):
        '''Root is depth 0'''
        return self.path.count(';')

####################################################################################################
    @property
    def parent_path(self):
        '''Used for `self.instances` lookup'''
        if self.path == ROOT_PATH:
            return None
        elif self.depth == 1:
            return ROOT_PATH
        elif self.depth > 1:
            return ';'.join(self.path.split(';')[:-2]) + ';'
        else:
            raise Exception()

    @property
    def parent_node(self):
        if self.path == ROOT_PATH:
            return None
        else:
            return self.instances[self.parent_path]

    @property
    def parent_taxid(self):
        if self.path == ROOT_PATH:
            return -1
        else:
            return self.instances[self.parent_path].taxid

####################################################################################################
    @property
    def lineage_paths(self) -> list:
        '''List of paths in lineage'''
        taxname_l = self.path.split(';')[:-1] # remove trailing '' because path is ';'-terminated
        lineage_path_l = []
        for i in range(1, self.depth+1):
            lineage_path_l.append(';'.join(taxname_l[:i]) + ';')

        return lineage_path_l


    @property
    def lineage_nodes(self):
        nodes = []
        for path in self.lineage_paths:
            nodes.append(self.instances[path])
        return nodes


    @property
    def lineage_ranks(self):
        ranks = []
        for node in self.lineage_nodes:
            ranks.append(node.rank)
        return ranks

####################################################################################################
    @property
    def taxon_name(self):
        if self.path == ROOT_PATH:
            return self.path

        return self.path.split(';')[-2]
        

    @property
    def mangled_taxon_name(self):
        if self.rankName_dup:
            return self.taxon_name + ' (taxid:%d)' % self.taxid
        else:
            return self.taxon_name

####################################################################################################
    def has_all_fixed_ranks(self):
        for rank in FIXED_RANKS:
            if rank not in self.lineage_ranks:
                return False
        return True

####################################################################################################
    @property
    def taxfile_entry(self):
        '''
        Must call identify_dups first
        '''
        return '*'.join([
            str(self.taxid), 
            self.mangled_taxon_name, 
            str(self.parent_taxid), 
            str(self.depth), 
            self.rank
        ])



####################################################################################################
    @property
    def full_mangled_path(self):
        taxname_l = []
        for path in self.lineage_paths:
            taxname_l.append(self.instances[path].mangled_taxon_name)
        
        return ';'.join([ROOT_PATH] + taxname_l)


####################################################################################################
    @classmethod
    def identify_dups(cls):
        rankName_l = []
        for node in cls.instances.values():
            rankName_l.append((node.rank, node.taxon_name))

        rankName_2_count = collapse_to_count(rankName_l)
        rankName_dup_l = [rankName for rankName, count in rankName_2_count.items() if count > 1]

        for node in cls.instances.values():
            rankName = (node.rank, node.taxon_name)
            if rankName in rankName_dup_l:
                node.rankName_dup = True
            else:
                node.rankName_dup = False

        # compensate for bug
        for node in cls.instances.values():
            if node.taxon_name == 'marine group' and node.rank == 'genus':
                node.rankName_dup = True

####################################################################################################
    @classmethod
    def inspect_lowest_ranks(cls, nodes=None):

        comment = 'all leaf nodes' if nodes is None else 'select leaf/internal nodes'

        if nodes is None: # inspect all leaves
            nodes = []
            for node in cls.instances.values():
                if node.is_leaf:
                    nodes.append(node)

        lineage_ranks_l = []
        last_rank_l = []
        for node in nodes:
            lineage_ranks = node.lineage_ranks
            lineage_ranks_l.append(';'.join(lineage_ranks))
            last_rank_l.append(lineage_ranks[-1])
        lineage_ranks_h = collapse_to_count(lineage_ranks_l)
        last_rank_h = collapse_to_count(last_rank_l)
        dprint(
            'last_rank_h # %s' % comment, 
            max_lines=None,
        )


####################################################################################################
    @classmethod
    def clear(cls):
        cls.instances.clear()
        cls.ranks.clear()






####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
def main(
    tax_fp,
    taxmap_fp,
    fasta_fp, 
    out_tax_fp,
    out_fasta_fp,
    rdp_clsf_jar
):
    """
----------------------------------------------------------------------------------------------------
 -t,--tax_file <arg>          contains the hierarchical taxonomy information in the following format:
                              taxid*taxon name*parent taxid*depth*rank
                              Fields taxid, the parent taxid and depth should be in integer format
                              The taxid, or the combination of taxon name and rank is unique
                              depth indicates the depth from the root taxon.
                              Note: the depth for the root is 0
----------------------------------------------------------------------------------------------------
 -s,--seq <arg>               training sequences in FASTA format with lineage in the header:
                              a list taxon names seperated by ';' with highest rank taxon first.
                              The lowest rank of the lineage have to be the same for all sequence.
                              The lowest rank is not limited to genus
----------------------------------------------------------------------------------------------------
    """

    logging.info('Parsing input taxon file')

    ###
    ### read taxonomy file
    df_tax = pd.read_csv(
        tax_fp, 
        sep='\t', 
        header=None, 
        names=['path', 'taxid', 'rank', 'remark', 'release'], 
        index_col=False
    )

    dprint("set(df_tax['rank'].tolist())",
        "sorted(df_tax['taxid'].tolist()[:30])",
        "max(list(df_tax['taxid']))",
        "[e for e in df_tax['remark'].tolist() if not (type(e) == float and np.isnan(e))] # non-nan remarks",
        "set(df_tax['release'].tolist())")

    df_tax.drop(['remark', 'release'], inplace=True, axis=1)
    
    dprint("df_tax # tax file")


    logging.info('Creating `Node` for each taxon path')

    ###
    ### create node for each taxon in taxonomy file
    Node(ROOT_PATH, 0, 'rootrank') # taxid=0 is not taken
    for index, row in df_tax.iterrows():
        Node(row['path'], row['taxid'], row['rank'])

    Node.identify_dups()

    Node.inspect_lowest_ranks()


    logging.info('Reading taxmap')

    ###
    ### read taxmap
    df_taxmap = pd.read_csv(
        taxmap_fp,
        sep='\t',
        header=0,
        index_col=False
    )
    df_taxmap.index = df_taxmap.apply(
        lambda row: '%s.%s.%s' % (row['primaryAccession'], row['start'], row['stop']), 
        axis=1
    )

    dprint('df_taxmap')

    ###
    ###
    def get_taxid_ctr():
        i = max(Node.taxid_l) + 100 # last taxid is 50000
        while True:
            yield i
            i += 1
    taxid_ctr = get_taxid_ctr()

    logging.info('Read input fasta file and writing output fasta file')
    
    ###
    ###
    prematures = []
    g = SeqIO.parse(fasta_fp, 'fasta')
    with open(out_fasta_fp, 'w') as fh_out:
        for record in g:
            
            acs_start_stop = record.id
            path = df_taxmap.loc[acs_start_stop, 'path']
            node = Node.instances[path]
            
            assert node.taxid == df_taxmap.loc[acs_start_stop, 'taxid']

            ###
            ###
            if node.rank == LOWEST_RANK:
                node = Node.instances[path]
            else:
                assert not node.has_all_fixed_ranks() # check not genus;major_clade;
                prematures.append(Node.instances[path])

                path = path + 'Genera Incertae Sedis;'
                if path in Node.instances:
                    node = Node.instances[path]
                else:
                    node = Node(path, next(taxid_ctr), LOWEST_RANK, rankName_dup=True)

            fh_out.write('>' + acs_start_stop + ' ' + node.full_mangled_path + '\n')
            fh_out.write(str(record.seq) + '\n')

    Node.inspect_lowest_ranks(prematures)

    logging.info('Iterating `Node`s and writing output taxon file')

    ###
    ###
    with open(out_tax_fp, 'w') as fh:
        for node in Node.instances.values():
            fh.write(node.taxfile_entry + '\n')

    #
    dprint('sorted(Node.taxid_l[:30])', run=globals())

    if rdp_clsf_jar is None:
        return

    ###
    ###



####################################################################################################
####################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate input files for RDP Classifier training"
        )
    )
    parser.add_argument(
        '--tax-file',
        required=True,
        help='SILVA taxonomy file (from SILVA)'
    )
    parser.add_argument(
        '--tax-map',
        required=True,
        help='SILVA ID to taxonomy mapping (from SILVA)'
    )
    parser.add_argument(
        '--fasta',
        required=True,
        help='SILVA ID to sequence FASTA (typically generated by QIIME2)'
    )
    parser.add_argument(
        '--out-tax-file',
        required=True,
        help='RDP Classifier training input file'
    )
    parser.add_argument(
        '--out-fasta',
        required=True,
        help='RDP Classifier training input file'
    )
    parser.add_argument(
        '--rdp-clsf-jar',
        required=False,
        help='RDP Classifier jar file'
    )

    return parser.parse_args()



if __name__ == '__main__':
    a = parse_args()


    main(
        a.tax_file,
        a.tax_map,
        a.fasta, 
        a.out_tax_file,
        a.out_fasta,
        a.rdp_clsf_jar
    )
