import sys
import math
from collections import namedtuple, defaultdict
from itertools import chain, product

START_SYM = 'ROOT'

class GrammarRule(namedtuple('Rule', ['lhs', 'rhs', 'log_prob'])):
    """A named tuple that represents a PCFG grammar rule.

    Each GrammarRule has three fields: lhs, rhs, log_prob

    Parameters
    ----------
    lhs : str
        A string that represents the left-hand-side symbol of the grammar rule.
    rhs : tuple of str
        A tuple that represents the right-hand-side symbols the grammar rule.
    log_prob : float
        The log probability of this rule.
    """
    def __repr__(self):
        return '{} -> {} [{}]'.format(
            self.lhs, ' '.join(self.rhs), self.log_prob)


def read_rules(grammar_filename):
    """Read PCFG grammar rules from grammar file

    The grammar file is a tab-separated file of three columns:
    probability, left-hand-side, right-hand-side.
    probability is a float number between 0 and 1. left-hand-side is a
    string token for a non-terminal symbol in the PCFG. right-hand-side
    is a space-delimited field for one or more  terminal and non-terminal
    tokens. For example:

        1	ROOT	EXPR
        0.333333	EXPR	EXPR + TERM

    Parameters
    ----------
    grammar_filename : str
        path to PCFG grammar file

    Returns
    -------
    set of GrammarRule
    """
    rules = set()
    with open(grammar_filename) as f:
        for rule in f.readlines():
            rule = rule.strip()
            log_prob, lhs, rhs = rule.split('\t')
            rhs = tuple(rhs.split(' '))
            assert rhs and rhs[0], rule
            rules.add(GrammarRule(lhs, rhs, math.log(float(log_prob))))
    return rules


class Grammar:
    """PCFG Grammar class."""
    def __init__(self, rules):
        """Construct a Grammar object from a set of rules.

        Parameters
        ----------
        rules : set of GrammarRule
            The set of grammar rules of this PCFG.
        """
        self.rules = rules

        self._rhs_rules = defaultdict(list)
        self._rhs_unary_rules = defaultdict(list)

        self._nonterm = set(rule.lhs for rule in rules)
        self._term = set(token for rhs in chain(rule.rhs for rule in rules)
                         for token in rhs if token not in self._nonterm)

        for rule in rules:
            _, rhs, _ = rule
            self._rhs_rules[rhs].append(rule)

        for rhs_rules in self._rhs_rules.values():
            rhs_rules.sort(key=lambda r: r.log_prob, reverse=True)

        self._is_cnf = all(len(rule.rhs) == 1
                           or (len(rule.rhs) == 2
                               and all(s in self._nonterm for s in rule.rhs))
                           for rule in self.rules)

    def from_rhs(self, rhs):
        """Look up rules that produce rhs

        Parameters
        ----------
        rhs : tuple of str
            The tuple that represents the rhs.

        Returns
        -------
        list of GrammarRules with matching rhs, ordered by their
        log probabilities in decreasing order.
        """
        return self._rhs_rules[rhs]

    def __repr__(self):
        summary = 'Grammar(Rules: {}, Term: {}, Non-term: {})\n'.format(
            len(self.rules), len(self.terminal), len(self.nonterminal)
        )
        summary += '\n'.join(sorted(self.rules))
        return summary

    @property
    def terminal(self):
        """Terminal tokens in this grammar."""
        return self._term

    @property
    def nonterminal(self):
        """Non-terminal tokens in this grammar."""
        return self._nonterm

    def get_cnf(self):
        """Convert PCFG to CNF and return it as a new grammar object."""
        nonterm = set(self.nonterminal)
        term = set(self.terminal)

        rules = list(self.rules)
        cnf = set()

        # STEP 1: eliminate nonsolitary terminals
        for i in range(len(rules)):
            rule = rules[i]
            lhs, rhs, log_prob = rule
            if len(rhs) > 1:
                rhs_list = list(rhs)
                for j in range(len(rhs_list)):
                    x = rhs_list[j]
                    if x in term:  # found nonsolitary terminal
                        new_nonterm = 'NT_{}'.format(x)
                        new_nonterm_rule = GrammarRule(new_nonterm, (x,), 0.0)

                        if new_nonterm not in nonterm:
                            nonterm.add(new_nonterm)
                            cnf.add(new_nonterm_rule)
                        else:
                            assert new_nonterm_rule in cnf
                        rhs_list[j] = new_nonterm
                rhs = tuple(rhs_list)
            rules[i] = GrammarRule(lhs, rhs, log_prob)

        # STEP 2: eliminate rhs with more than 2 nonterminals
        for i in range(len(rules)):
            rule = rules[i]
            lhs, rhs, log_prob = rule
            if len(rhs) > 2:
                assert all(x in nonterm for x in rhs), rule
                current_lhs = lhs
                for j in range(len(rhs) - 2):
                    new_nonterm = 'BIN_"{}"_{}'.format(
                        '{}->{}'.format(lhs, ','.join(rhs)), str(j))
                    assert new_nonterm not in nonterm, rule
                    nonterm.add(new_nonterm)
                    cnf.add(
                        GrammarRule(current_lhs,
                                    (rhs[j], new_nonterm),
                                    log_prob if j == 0 else 0.0))
                    current_lhs = new_nonterm
                cnf.add(GrammarRule(current_lhs, (rhs[-2], rhs[-1]), 0.0))
            else:
                cnf.add(rule)

        return Grammar(cnf)

            
    def parse(self, line):
        """Parse a sentence with the current grammar.

        The grammar object must be in the Chomsky normal form.

        Parameters
        ----------
        line : str
            Space-delimited tokens of a sentence.
        """
        
        #Input: Binarized PCFG G = (Σ, N, R, S), a string x = x1 . . . xn
        line = line.split()
        if len(line) == 0:
            return
        nontermlist = list(self._nonterm)
        
        #initialization
        bestScore = [[{} for j in range(len(line)+1)] for i in range(len(line))]
        backptr = [[{} for j in range(len(line)+1)] for i in range(len(line))]
        
        #Fill terminal rules
        for i in range(len(line)):
            for A in nontermlist:
                for r_rules in self.from_rhs((line[i],)): #where (p, A → xi) ∈ R
                    if r_rules.lhs == A:
                        p = r_rules.log_prob
                        if A in bestScore[i][i+1]:
                            if bestScore[i][i+1][A] < p:
                                bestScore[i][i+1][A] = p
                                backptr[i][i+1][A] = (None, None, line[i], p)
                        else:
                            bestScore[i][i+1][A] = p
                            backptr[i][i+1][A] = (None, None, line[i], p)
              
      
            #Add unary rules for cell [i-1, i]
            stack = []
            for cur in bestScore[i][i+1]:
                stack.append(cur)
            while len(stack):
                A = stack.pop()
                unarylist = self.from_rhs((A,))
                for ul in unarylist:
                    B = ul.lhs
                    p = ul.log_prob
                    #print(A, B)
                    if B in bestScore[i][i+1]:
                        if bestScore[i][i+1][B] < p + bestScore[i][i+1][A]:
                            bestScore[i][i+1][B] = p + bestScore[i][i+1][A]
                            backptr[i][i+1][B] = (None, None, A, p)
                            stack.append(B)
                    else:
                        bestScore[i][i+1][B] = p + bestScore[i][i+1][A]
                        backptr[i][i+1][B] = (None, None, A, p)

                        stack.append(B)
                
                
        # l: span length, i: start, j: end, k: split    
        for l in range(2, len(line) + 1):
            for i in range(len(line) - l + 1):
                j = i + l
                for k in range(i+1, j):
                    for rule in self.rules:
                        if len(rule.rhs) == 2:
                            A = rule.lhs
                            B, C = rule.rhs
                            p = rule.log_prob
                            if B in bestScore[i][k] and C in bestScore[k][j]:
                                if A in bestScore[i][j]:
                                    if bestScore[i][j][A] < p + bestScore[i][k][B] + bestScore[k][j][C]:
                                        bestScore[i][j][A] = p + bestScore[i][k][B] + bestScore[k][j][C]
                                        backptr[i][j][A] = (k, B, C, p)
                                else:
                                    bestScore[i][j][A] = p + bestScore[i][k][B] + bestScore[k][j][C]
                                    backptr[i][j][A] = (k, B, C, p)
                
                #Add unary rules for non-terminals in cell [i, j]
                for cur in bestScore[i][j]:
                    stack.append(cur)    
                while len(stack):
                    A = stack.pop()
                    unarylist = self.from_rhs((A,))
                    for ul in unarylist:
                        B = ul.lhs
                        p = ul.log_prob
                        if B in bestScore[i][j]:
                            if bestScore[i][j][B] < p + bestScore[i][j][A]:
                                bestScore[i][j][B] = p + bestScore[i][j][A]
                                backptr[i][j][B] = (None, None, A, p)
                                stack.append(B)
                        else:
                            bestScore[i][j][B] = p + bestScore[i][j][A]
                            backptr[i][j][B] = (None, None, A, p)
                            stack.append(B)
                        
                                      
#         for i in range(len(line)):
#             for j in range(i, len(line)+1):
#                 print(i,j,backptr[i][j], end=" ")
#             print()
                      
        #reconstruct the parse tree 
        if START_SYM in bestScore[0][len(line)]:
            print(self.printTree(bestScore, backptr, 0, len(line), START_SYM, 0))
            print(bestScore[0][len(line)][START_SYM])
        else:
            print("NONE")
        
    def printTree(self, bestScore, backptr, start, end, symb, indent):
        
        if symb not in self._nonterm:
            return symb
        
        split, le, re, p = backptr[start][end][symb]
        
        if split is None and le is None:
            new = indent + len(symb) + 2
            out = self.printTree(bestScore, backptr, start, end, re, new)
            if p == 0 and symb != START_SYM:
                return out
            else:
                return "(" + symb + " " + out + ")"
                
        else:
            if p == 0:
                left = self.printTree(bestScore, backptr, start, split, le, indent)
                right = self.printTree(bestScore, backptr, split, end, re, indent)
                return left + '\n' + ' '*indent + right
            else:
                new = indent + len(symb) + 2 
                left = self.printTree(bestScore, backptr, start, split, le, new)
                right = self.printTree(bestScore, backptr, split, end, re, new)
                return '(' + symb + ' ' + left + '\n' + ' '*new + right + ')'

