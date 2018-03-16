"""
Goal of this module is to provide class that can generate web-scraper based on training sample using decision trees (here called splits).

Contents:
FinStat and SplitStat -- classes for statistics about leaf or split
xSplit, xClassifier -- pair of classes for adding classification
Scraper -- resulting scraper
AutoScraper -- class for creating training scrapers

xSplit should provide following functionality (it is synonimous to 'feature'):
# contain full information about split-decision
# apply(self, features):
## applied to only 1 object, but receives full feature-set from respective classifier
## if received None should return None
## should return True or False otherwise

xClassifier should provide following functionality:
# get_splits(self): return list of all possible splits
# get_features(self, sample): sample is list or tuple or element, should return features for each 
# get_features_on_splits(self, splits): same, but calculates only 'active' features; is splits are None or [], returns None
"""


from lxml import etree
from lxml import html
from collections import Counter
import numpy as np

class FinStat:
    def __init__(self, n0, n1):
        self.__n0 = n0
        self.__n1 = n1
        
    def __repr__(self):
        return '0:{} 1:{}'.format(self.__n0, self.__n1)
    
    def total(self):
        return self.__n0 + self.__n1
    
    def mode(self):
        return 1 if self.__n1 > self.__n0 else 0
    
    def single(self, default=None):
        if self.__n0 == 0 and self.__n1 == 0:
            return default
        if self.__n0 == 0:
            return 1
        if self.__n1 == 0:
            return 0
        return default
    
    def mean(self):
        return self.__n1 / (self.__n0 + self.__n1)

class SplitStat:
    def __init__(self, num_true0, num_true1, num_false0, num_false1):
        self.__nt0 = num_true0
        self.__nt1 = num_true1
        self.__nf0 = num_false0
        self.__nf1 = num_false1
        
    def __repr__(self):
        return '{}-{} vs {}-{}'.format(self.__nf0, self.__nf1, self.__nt0, self.__nt1)
    
    def get_stat(self, mask):
        if mask==True:
            return FinStat(self.__nt0, self.__nt1)
        elif mask==False:
            return FinStat(self.__nf0, self.__nf1)
        return None
    
    def get_total_stat(self):
        return FinStat(self.__nt0+self.__nf0, self.__nt1 + self.__nf1)
        
    def calc_from_two_sets(targets_true, targets_false):
        nt1 = sum(targets_true)
        nt0 = len(targets_true) - nt1
        nf1 = sum(targets_false)
        nf0 = len(targets_false) - nf1
        return SplitStat(nt0, nt1, nf0, nf1)
    
    def calc_from_masked(targets, mask):
        return SplitStat.calc_from_two_sets([targets[i] for i in range(len(targets)) if mask[i]],
                                         [targets[i] for i in range(len(targets)) if not mask[i]])
        
    def total_num(self):
        return self.__nt0 + self.__nt1 + self.__nf0 + self.__nf1
    
    def total_0(self):
        return self.__nt0 + self.__nf0
    
    def total_1(self):
        return self.__nt1 + self.__nf1
    
    def total_true(self):
        return self.__nt0 + self.__nt1
    
    def total_false(self):
        return self.__nf0 + self.__nf1
    
    def true_0(self):
        return self.__nt0
    
    def true_1(self):
        return self.__nt1
    
    def false_0(self):
        return self.__nf0
    def false_1(self):
        return self.__nf1
    
    def calc_gini(clsf_name, split, split_stat):
        #p0l = l0 / (l0 + r0)
        #p0r = r0 / (l0 + r0)
        #p1l = l1 / (l1 + r1)
        #p1r = r1 / (l1 + r1)

        p0l = split_stat.false_0() / split_stat.total_0()
        p0r = split_stat.true_0() / split_stat.total_0()
        p1l = split_stat.false_1() / split_stat.total_1()
        p1r = split_stat.true_1() / split_stat.total_1()

        gini = 2*(p0l * p1l * 0.5 + p0r * 0.5 * (1 + p1l) - 0.5)
        return abs(gini)
    
class TagSplit:
    def __init__(self, tag_id_value, tag_name):
        self.__idx = tag_id_value
        self.__tag_name = tag_name
        
    def __repr__(self):
        return '<{}>'.format(self.__tag_name)
        
    def idx(self):
        return self.__idx
    
    def tag_name(self):
        return self.__tag_name
    
    def apply(self, features):
        return (features == self.__idx)
    
class AttribSplit:
    def __init__(self, attrkv_id_value, key_name, value_name):
        self.__idx = attrkv_id_value
        self.__key_name = key_name
        self.__value_name = value_name
    
    def __repr__(self):
        return '{}={}'.format(self.__key_name, self.__value_name)
        
    def idx(self):
        return self.__idx
    
    def key_name(self):
        return self.__key_name
    
    def value_name(self):
        return self.__value_name
    
    def apply(self, features):
        if features is None:
            return None
        return (self.__idx in features)

class TagClassifier:
    def __init__(self, used_tags, use_other=True):
        self.__tag2id = {x:i for (i, x) in enumerate(used_tags)}
        self.__id2tag = {i:x for (i, x) in enumerate(used_tags)}
        self.__use_other = use_other
        
    def tag2id(self, tag):
        if tag is None:
            return None
        if tag in self.__tag2id:
            return self.__tag2id[tag]
        return None if not self.__use_other else -1
    
    def id2tag(self, idx):
        if idx is None:
            return 'null'
        if idx == -1 and self.__use_other:
            return 'other'
        return self.__id2tag[idx] if idx in self.__id2tag else None
    
    def get_features(self, sample):
        if type(sample) is list:
            return [self.get_features(z) for z in sample]
        if type(sample) is tuple:
            return (self.get_features(z) for z in sample)
        if sample is None:
            return None
        return self.tag2id(sample.tag)
    
    
    def get_features_on_splits(self, sample, splits):
        if splits is None or splits == []:
            return None
        return self.get_features(sample)
    
    def get_splits(self):
        values = [None] + list(range(len(self.__tag2id)))
        if self.__use_other:
            values.append(-1)
        return [TagSplit(v, self.id2tag(v)) for v in values]
    
class AttribClassifier:
    def __init__(self, used_attrib_keyvalues):
        self.__akv2id = {p:i for (i, p) in enumerate(used_attrib_keyvalues)}
        self.__id2akv = {i:p for (i, p) in enumerate(used_attrib_keyvalues)}
        
    def akv2id(self, akv):
        return self.__akv2id[akv] if akv in self.__akv2id else None
    
    def id2akv(self, idx):
        return self.__id2akv[idx] if idx in self.__id2akv else None
    
    def get_features(self, sample):
        if type(sample) is list:
            return [self.get_features(z) for z in sample]
        if type(sample) is tuple:
            return (self.get_features(z) for z in sample)
        
        kvids = []
        if sample is None:
            return kvids
        for kv in sample.attrib.items():
            if kv in self.__akv2id:
                kvids.append(self.__akv2id[kv])
        return kvids
    
    def get_features_on_splits(self, sample, splits):
        if splits is None or splits == []:
            return None
        return self.get_features(sample)
    
    def get_splits(self):
        values = list(range(len(self.__akv2id)))
        return [AttribSplit(v, self.__id2akv[v][0], self.__id2akv[v][1]) for v in values]

class IdxSplit:
    def __init__(self, base_split, idx):
        self.__base = base_split
        self.__idx = idx
        
    def __repr__(self):
        return '{} at id={}'.format(self.__base.__repr__(), self.__idx)
    
    def idx(self):
        return self.__idx
    
    def base(self):
        return self.__base
        
    def apply(self, features):
        if features is None:
            return None
        return self.__base.apply(features[self.__idx])
    
class ContextSplit(IdxSplit):
    def __init__(self, base_split, idx, name):
        IdxSplit.__init__(self, base_split, idx)
        self.__name = name
        
    def __repr__(self):
        return '{} at context {}'.format(self.base().__repr__(), self.__name)
        
    def name(self):
        return self.__name
    
class ParentAbsSplit(IdxSplit):
    def __init__(self, base_split, idx):
        IdxSplit.__init__(self, base_split, idx)
        
    def __repr__(self):
        return '{} at abs-depth={}'.format(self.base().__repr__(), self.idx())
    

class ParentRelSplit(IdxSplit):
    def __init__(self, base_split, idx):
        IdxSplit.__init__(self, base_split, idx)
        
    def __repr__(self):
        return '{} at rel-depth={}'.format(self.base().__repr__(), self.idx())

class FwdIterSplit(IdxSplit):
    def __init__(self, base_split, idx):
        IdxSplit.__init__(self, base_split, idx)
        
    def __repr__(self):
        return '{} at fwd-iter-id={}'.format(self.base().__repr__(), self.idx())


class ContextClassifier:
    def __init__(self, base_classifier, used_context):
        self.__base = base_classifier
        self.__ctxt = used_context
        self.__name2ctxt = {x[0]:(i, x[1]) for (i, x) in enumerate(used_context)}
        self.__id2ctxt = {i:x for (i, x) in enumerate(used_context)}
        
    def id2ctxt(idx):
        return self.__id2ctxt[idx] if idx in self.__id2ctxt else None
    
    def name2ctxt(name): 
        return self.__name2ctxt[name] if name in self.__name2ctxt else None
    
    def get_features(self, sample):
        if type(sample) is list:
            return [self.get_features(z) for z in sample]
        if type(sample) is tuple:
            return (self.get_features(z) for z in sample)
        
        return [self.__base.get_features(func(sample)) for (_,func) in self.__ctxt]
    
    def get_features_on_splits(self, sample, splits):
        if splits is None or splits == []:
            return None

        if type(sample) is list:
            return [self.get_features_on_splits(z, splits) for z in sample]
        if type(sample) is tuple:
            return (self.get_features_on_splits(z, splits) for z in sample)
        
        return [self.__base.get_features_on_splits(func(sample), [split.base() for split in splits if split.idx()==i])
                for (i,(_,func)) in enumerate(self.__ctxt)]
    
    def get_splits(self):
        bs = self.__base.get_splits()
        return [ContextSplit(b, i, name) for (i, (name,_)) in enumerate(self.__ctxt) for b in bs]

class ParentAbsClassifier:
    def __init__(self, base_classifier, max_depth):
        self.__base = base_classifier
        self.__depth = max_depth
        
    def max_depth():
        return self.__depth
    
    def get_features(self, sample):
        if type(sample) is list:
            return [self.get_features(z) for z in sample]
        if type(sample) is tuple:
            return (self.get_features(z) for z in sample)
        
        res = []
        p = sample
        empty_elem = self.__base.get_features(None)
        while p is not None:
            res.append(self.__base.get_features(p))
            p = p.getparent()

        return list(reversed(res[:self.__depth])) + [empty_elem]*max(0, self.__depth - len(res))
    
    def get_features_on_splits(self, sample, splits):
        if splits is None or splits == []:
            return None
        
        if type(sample) is list:
            return [self.get_features_on_splits(z, splits) for z in sample]
        if type(sample) is tuple:
            return (self.get_features_on_splits(z, splits) for z in sample)
        
        res = []
        p = sample
        while p is not None:
            res.append(p)
            p = p.getparent()
        
        res = list(reversed(res[:self.__depth])) + [None]*max(0, self.__depth - len(res))

        return [self.__base.get_features_on_splits(p, [split.base() for split in splits if split.idx()==i])
                for (i,p) in enumerate(res)]
    
    def get_splits(self):
        bs = self.__base.get_splits()
        return [ParentAbsSplit(b, i) for i in range(self.__depth) for b in bs]


class ParentRelClassifier:
    def __init__(self, base_classifier, max_depth):
        self.__base = base_classifier
        self.__depth = max_depth
        
    def max_depth():
        return self.__depth
    
    def get_features(self, sample):
        if type(sample) is list:
            return [self.get_features(z) for z in sample]
        if type(sample) is tuple:
            return (self.get_features(z) for z in sample)
        
        res = []
        p = sample
        empty_elem = self.__base.get_features(None)
        while p is not None:
            res.append(self.__base.get_features(p))
            p = p.getparent()

        return res[:self.__depth] + [empty_elem]*max(0, self.__depth - len(res))
    
    def get_features_on_splits(self, sample, splits):
        if splits is None or splits == []:
            return None
        
        if type(sample) is list:
            return [self.get_features_on_splits(z, splits) for z in sample]
        if type(sample) is tuple:
            return (self.get_features_on_splits(z, splits) for z in sample)
        
        res = []
        p = sample
        while p is not None:
            res.append(p)
            p = p.getparent()
        
        res = res[:self.__depth] + [None]*max(0, self.__depth - len(res))

        return [self.__base.get_features_on_splits(p, [split.base() for split in splits if split.idx()==i])
                for (i,p) in enumerate(res)]
    
    def get_splits(self):
        bs = self.__base.get_splits()
        return [ParentRelSplit(b, i) for i in range(self.__depth) for b in bs]
    
    

class FwdIterClassifier:
    def __init__(self, base_classifier, max_num):
        self.__base = base_classifier
        self.__num = max_num
        
    def max_num():
        return self.__num
    
    def get_features(self, sample):
        if type(sample) is list:
            return [self.get_features(z) for z in sample]
        if type(sample) is tuple:
            return (self.get_features(z) for z in sample)
        
        res = []
        empty_elem = self.__base.get_features(None)
        i = 0
        for p in sample.iter():
            if p == sample:
                continue
            res.append(self.__base.get_features(p))
            i += 1
            if i >= self.__num:
                break
                
        return res[:self.__num] + [empty_elem]*max(0, self.__num - len(res))
    
    def get_features_on_splits(self, sample, splits):
        if splits is None or splits == []:
            return None
        
        if type(sample) is list:
            return [self.get_features_on_splits(z, splits) for z in sample]
        if type(sample) is tuple:
            return (self.get_features_on_splits(z, splits) for z in sample)
        
        res = []
        i = 0
        for p in sample.iter():
            if p == sample:
                continue
            res.append(p)
            i += 1
            if i >= self.__num:
                break
        
        res = res[:self.__num] + [None]*max(0, self.__num - len(res))

        return [self.__base.get_features_on_splits(p, [split.base() for split in splits if split.idx()==i])
                for (i,p) in enumerate(res)]
    
    def get_splits(self):
        bs = self.__base.get_splits()
        return [FwdIterSplit(b, i) for i in range(self.__num) for b in bs]

class Scraper:
    def gather_list_from_tree(t):
        if t is None:
            return []
        return [t[0]] + Scraper.gather_list_from_tree(t[1]) + Scraper.gather_list_from_tree(t[2])
    
    def gather_sample_from_root(r):
        res = []
        for x in r.iter():
            res.append(x)
        return res
    
    def apply_tree_split(dict_features, split_tree, stat):
        if split_tree is None:
            return [stat] * max([len(v) for (k,v) in dict_features.items()])
        
        split = split_tree[0]
        tsplit = split_tree[1]
        fsplit = split_tree[2]
        
        features = dict_features[split[0]]
        if features == []:
            return [stat] * max([len(v) for (k,v) in dict_features.items()])
        
        mask = [split[1].apply(x) for x in features]
        
        
        td_features = {key:[f for (i, f) in enumerate(features) if mask[i] == True]
                       for (key, features) in dict_features.items() if features is not None}
        fd_features = {key:[f for (i, f) in enumerate(features) if mask[i] == False]
                       for (key, features) in dict_features.items() if features is not None}
        
        tres = Scraper.apply_tree_split(td_features, tsplit, split[2].get_stat(True))
        fres = Scraper.apply_tree_split(fd_features, fsplit, split[2].get_stat(False))
        
        ct = np.cumsum(np.array(mask))-1
        cf = np.cumsum(~np.array(mask))-1
        
        return [tres[ct[i]] if v else fres[cf[i]] for (i,v) in enumerate(mask)]
    
    def __init__(self, dict_classifiers, split_tree):
        self.__classifiers = dict_classifiers
        self.__split_tree = split_tree
        self.__split_list = Scraper.gather_list_from_tree(split_tree)
        
    def get_split_tree(self):
        return self.__split_tree
    
    def get_stats(self, sample):
        d_features = {key:clsf.get_features_on_splits(sample, [split[1] for split in self.__split_list if split[0] == key])
                        for (key,clsf) in self.__classifiers.items()}
        return Scraper.apply_tree_split(d_features, self.__split_tree, self.__split_tree[0][2].get_total_stat())
    
    def proba(self, x):
        stats = []
        if type(x) is list:
            stats = self.get_stats(x)
        elif type(x) is etree._ElementTree:
            stats = self.get_stats(Scraper.gather_sample_from_root(x))
        return [z.mean() for z in stats]
    
    def predict(self, x):
        stats = []
        if type(x) is list:
            stats = self.get_stats(x)
        elif type(x) is etree._ElementTree:
            stats = self.get_stats(Scraper.gather_sample_from_root(x))
        return [z.single(default=False) for z in stats]
    
    def select(self, x):
        stats = []
        sample = None
        if type(x) is list:
            sample = x
            stats = self.get_stats(x)
        elif type(x) is etree._ElementTree:
            sample = Scraper.gather_sample_from_root(x)
            stats = self.get_stats(sample)
        return [sample[i] for (i,z) in enumerate(stats) if z.single(default=False)]

class ScraperNode:
    def __init__(self, name, elem, subvals):
        self.__name = name
        self.__elem = elem
        self.__subvals = {}
        if subvals is not None and type(subvals) is dict:
            self.__subvals = subvals
        elif subvals is not None and type(subvals) is list and len(subvals) > 0:
            vn = set(v.name() for v in subvals)
            self.__subvals = {k: [v for v in subvals if v.name()==k] for k in vn}
        
    def __repr__(self):
        if self.is_final():
            return '{0}({1})'.format(self.__name, self.__elem)
        return '{0}({1}) with {2}'.format(self.__name, self.__elem, self.get_all())
        
    def renamed(self, name):
        return ScraperNode(name, self.__elem, self.__subvals)
        
    def is_final(self):
        return len(self.__subvals) == 0
    
    def get_num(self, subname):
        if subname not in self.__subvals:
            return 0
        return len(self.__subvals[subname])
    
    def get_list(self, subname):
        if subname not in self.__subvals:
            return []
        return self.__subvals[subname]
    
    def get(self, subname):
        if subname not in self.__subvals:
            return None
        res = self.__subvals[subname]
        if len(res) == 0:
            return None
        if len(res) == 1:
            return res[0]
        return res
    
    def get_all(self):
        return [x for (k, v) in self.__subvals.items() for x in v]
    
    def __getitem__(self, subname):
        return self.get(subname)
    
    def elem(self):
        return self.__elem
        
    def name(self):
        return self.__name

    
class AutoScraper:
    def __init__(self, dict_classifiers):
        self.__classifiers = dict_classifiers
            
    def get_features(self, sample):
        return {key:clsf.get_features(sample) for (key, clsf) in self.__classifiers.items()}
    
    def get_splits(self):
        return {key:clsf.get_splits() for (key, clsf) in self.__classifiers.items()}
    
    def gather_splits(self, targets, dict_features, dict_splits):
        all_splits = []
        for (clsf_name, features) in dict_features.items():
            splits = dict_splits[clsf_name]
            for split in splits:
                res = [split.apply(x) for x in features]
                all_splits.append((clsf_name, split, SplitStat.calc_from_masked(targets, res)))
        return all_splits
    
    def select_split(self, splits, order_function=SplitStat.calc_gini):
        return list(sorted(splits, key=lambda x:order_function(*x), reverse=True))[0]
    
    def apply_split(self, dict_features, split):
        features = dict_features[split[0]]
        return [split[1].apply(x) for x in features]
    
    def run_one_step(self, targets, dict_features, dict_splits, order_function=SplitStat.calc_gini):
        all_splits = self.gather_splits(targets, dict_features, dict_splits)
        split = self.select_split(all_splits, order_function)
        mask = self.apply_split(dict_features, split)
        
        #true:
        t_targets = [targets[i] for i in range(len(targets)) if mask[i]]
        t_features = {key:[features[i] for i in range(len(targets)) if mask[i]] for (key, features) in dict_features.items()}
        
        #false:
        f_targets = [targets[i] for i in range(len(targets)) if not mask[i]]
        f_features = {key:[features[i] for i in range(len(targets)) if not mask[i]] for (key, features) in dict_features.items()}
        
        return (split, (t_targets, t_features), (f_targets, f_features))
    
    def run_n_steps(self, targets, dict_features, dict_splits, max_depth, order_function=SplitStat.calc_gini):
        if max_depth == 0:
            return None
        if len(targets) == 0:
            return None
        if sum(targets) == 0 or sum(targets) == len(targets):
            return None
        
        (split, (t_targets, t_features), (f_targets, f_features)) = self.run_one_step(targets, dict_features, dict_splits, order_function)
        t_res = self.run_n_steps(t_targets, t_features, dict_splits, max_depth-1, order_function)
        f_res = self.run_n_steps(f_targets, f_features, dict_splits, max_depth-1, order_function)
        return (split, t_res, f_res)
    
    def get_features_on_splits(self, sample, splits):
        return {key:clsf.get_features_on_splits(sample, [split[1] for split in splits if split[0] == key])
                    for (key,clsf) in self.__classifiers.items()}
    
    def fit_on_sample(self, sample, targets, max_depth=5, order_function=SplitStat.calc_gini):
        d_features = self.get_features(sample)
        d_splits = self.get_splits()
        split_tree = self.run_n_steps(targets, d_features, d_splits, max_depth, order_function)
        return Scraper(self.__classifiers, split_tree)
    
    def fit(self, x, y, max_depth=5, order_function=SplitStat.calc_gini):
        if type(x) is etree._ElementTree:
            sample = []
            targets = []
            for z in x.iter():
                sample.append(z)
                targets.append(z in y)
            return self.fit_on_sample(sample, targets, max_depth, order_function)
        if len(x) != len(y) and len(y) > 0 and type(y[0]) is html.HtmlElement:
            targets = [(z in y) for z in x]
            return self.fit_on_sample(x, targets, max_depth, order_function)
        if len(x) == len(y) and type(y[0]) is int:
            return self.fit_on_sample(x, y, max_depth, order_function)
        raise Exception('Unsupported (x, y) combination')
    
    def transform_into_tree(data):
        """
        Transforms list of form [(node, meta),(node, meta)] into list of lists
        """
        lev = [(x, meta, x) for (x, meta) in data]
        while True:
            nlev = [(x, meta, (z.getparent() if z is not None else None)) for (x, meta, z) in lev]
            
            
            lev = nlev
            if len([z for (_,_,z) in lev if z is not None]) == 0:
                break
                     
            
        
    def raw_stats(doc_tree):
        tag_stat = []
        attrib_stat = []
        depth_stat = []
        attrv_stat = {}
        for x in doc_tree.iter():
            tag_stat.append(x.tag)
            attrib_stat += x.attrib.keys()
            
            p = x.getparent()
            d = 0
            while p is not None:
                d += 1
                p = p.getparent()
            
            depth_stat.append(d)
            for k in x.attrib:
                if k not in attrv_stat:
                    attrv_stat[k] = []
                attrv_stat[k].append(x.attrib[k])
        return depth_stat, tag_stat, attrib_stat, attrv_stat
    
    def get_stats(doc_tree):
        depth_stat, tag_stat, attrib_stat, attrv_stat = AutoScraper.raw_stats(doc_tree)
        depth_info = list(sorted(dict(Counter(depth_stat)).items(), key=lambda x: x[0]))
        tag_info = list(sorted(dict(Counter(tag_stat)).items(), key=lambda x: x[1], reverse=True))
        attrk_info = list(sorted(dict(Counter(attrib_stat)).items(), key=lambda x: x[1], reverse=True))
        attr_all_vals = [y for x in attrv_stat.values() for y in x]
        attrv_info = list(sorted(dict(Counter(attr_all_vals)).items(), key=lambda x: x[1], reverse=True))
        attrkv_info = {k:list(sorted(dict(Counter(v)).items(), key=lambda x: x[1], reverse=True))
                       for (k, v) in attrv_stat.items()}
        return depth_info, tag_info, attrk_info, attrv_info, attrkv_info

    def get_child(x, idx):
        if x is None:
            return None
        cs = x.getchildren()
        if cs is None or len(cs) == 0:
            return None
        if idx >= 0:
            if idx < len(cs):
                return cs[idx]
        else:
            if -idx-1 < len(cs):
                return cs[idx]
        return None

    def get_sibling(x, idx):
        if x is None:
            return None
        p = x
        i = idx
        while i != 0:
            if i > 0:
                i-=1
                p = p.getnext()
            else:
                i+=1
                p = p.getprevious()
            if p is None:
                return None
        return p
    
class MultiScraper:
    def __init__(self, dict_scrapers, dict_structure):
        self.__scrapers = dict_scrapers
        self.__strdef = dict_structure
        
    def parse(self, x):
        res0 = [(z, k) for (k,v) in self.__scrapers.items() for z in v.select(x)]
        res1 = MultiScraper.induce_hierarchy(res0)
        res = MultiScraper.induce_structure(res1, self.__strdef)
        return res
    
    def parse_(self, x):
        res0 = [(z, k) for (k,v) in self.__scrapers.items() for z in v.select(x)]
        print(len(res0))
        res = MultiScraper.induce_hierarchy(res0)
        return res
    
    def get_full_path(x):
        res = []
        p = x
        while p is not None:
            res.append(p)
            p = p.getparent()
        return res
    
    def induce_hierarchy_step(lst):
        r = [(p[-1] if len(p)>0 else None) for (_,_,p) in lst] #get last parents
        s = set(r) #get unique, but ruins order
        sd = {k:min([i for (i,(_,_,p)) in enumerate(lst) if (p[-1] if len(p)>0 else None)==k]) for k in s} #get first position
        sl = [x for (x,_) in sorted(sd.items(), key=lambda x: x[1])] #restore order

        if len(s) == 1:
            #either final node or pass-through
            if None in s:
                assert(len(lst) == 1)
                return (lst[0][0],lst[0][1],[]) #final
            return MultiScraper.induce_hierarchy_step([(x,m,p[:-1]) for (x,m,p) in lst]) #pass-through
        if None in s:
            print('hit none in MultiScraper.induce_hierarchy_step - check adequacy')
            #check that num of None is exactly 1
            xs = [(x,m) for (x,m,p) in lst if len(p)==0]
            assert(len(xs)==1)
            (x0, m0) = xs[0]
            res = [MultiScraper.induce_hierarchy_step([(x,m,p[:-1]) for (x,m,p) in lst if len(p)>0 and p[-1]==v])
                   for v in [x for x in sl if x is not None]]
            return (x0, m0, res) 
        #standard branch
        res = [MultiScraper.induce_hierarchy_step([(x,m,p[:-1]) for (x,m,p) in lst if p[-1]==v]) for v in sl]
        #print('\nres=',res)
        #print(lst[0])
        return (lst[0][2][-1].getparent(), None, res)


    def induce_hierarchy(lst):
        #all nodes shall be distinct
        lev = [(x, d, MultiScraper.get_full_path(x)) for (x,d) in lst]
        return MultiScraper.induce_hierarchy_step(lev)
    
    def check_substructure(lst, structure_descr):
        actual = dict(Counter([v.name() for v in lst]))
        #print('test: ', actual, structure_descr)
        if None in actual:
            return False
        if 'other' in structure_descr:
            actual_other = sum([v for (k,v) in actual.items() if k not in structure_descr])
            constr_other = structure_descr['other']
            if actual_other < constr_other[0] or (len(constr_other) > 1 and actual_other > constr_other[1]):
                return False
        res1 = [(k in actual and actual[k] >= v[0] and (len(v) == 1 or actual[k] <= v[1]))
                    for (k,v) in structure_descr.items() if k != 'other' and v[0] > 0]
        res2 = [(len(v) == 1 or actual[k] <= v[1]) for (k,v) in structure_descr.items() if k != 'other' and v[0] == 0 and k in actual]
        return all(res1) and all(res2)

    def induce_structure(node, struct_transform):
        if node[2] == []:
            return ScraperNode(node[1], node[0], [])
        res = [MultiScraper.induce_structure(n, struct_transform) for n in node[2]]
        strhit = [t for (d, t) in struct_transform if MultiScraper.check_substructure(res, d)]
        if len(strhit) > 1:
            raise 'Multiple structures matched!'
        fin = ScraperNode(None, node[0], res)
        if len(strhit) == 1:
            transform = strhit[0]
            if type(transform) is str:
                fin = fin.renamed(transform)
            else:
                #print('applying transform', strhit[0])
                fin = strhit[0](fin)
        return fin