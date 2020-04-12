"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for key in self.lhs_to_rules:
            lhs_rule=self.lhs_to_rules[key]
            all_pro=[]
            for lhs in lhs_rule:
                all_pro.append(lhs[2])
                if len(lhs[1])==2 and (lhs[1][0].islower() or lhs[1][1].islower()):
                    return False
                elif len(lhs[1])==1 and (lhs[1][0].isupper() or lhs[1][0].isupper()):
                    return False
                else:
                    continue
            if round(fsum(all_pro),2)!=1:
                return False
        return True


if __name__ == "__main__":
    # with open(sys.argv[1],'r') as grammar_file:
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
    print(grammar.lhs_to_rules)
    print(grammar.lhs_to_rules['ADJP'])
    print(grammar.lhs_to_rules['ADJP'][0][1])
    print(grammar.verify_grammar())