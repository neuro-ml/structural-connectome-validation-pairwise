from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from collections import OrderedDict
from itertools import product
from pickle import dump, load
from pprint import pprint, pformat
from pandas import DataFrame
from numpy import mean, std


class Dict(OrderedDict):
    '''
    Ordered dictionary. 

    Parameters
    ----------
    items: list
        List of tuples, where each element is tuple of key and value
    '''
    
    def __init__(self, items):
        super(Dict, self).__init__(items)
        for key in self.keys():
            if type(self[key]) == list:
                self[key] = Dict(self[key])

    def __getattr__(self, attr):
        if not attr.startswith('_'):
            return self[attr]
        super(Dict, self).__getattr__(attr)
    
    def __setattr__(self, key, value):
        if not key.startswith('_'):
            self.__setitem__(key, value)
        else:
            super(Dict, self).__setattr__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]

    def __iadd__(self, tuples):
        if type(tuples) == list:
            tuples = Dict(tuples)
        elif type(tuples) == tuple:
            tuples = Dict([tuples])
        for key in tuples.keys():
            self[key] = tuples[key]
        return self

    # add addition and substraction

    def get(self, *keys):
        value = self
        for key in keys:
            value = value[key]
        return value

class Steps(Dict):
    '''
    Making comfortable interface for work with steps.
    '''
    def __init__(self, items):
        super(Steps, self).__init__(items)

    def _struct_transform(self, input_list):
        structure = []
        if type(input_list) == tuple:
            input_list = [input_list]
        for key, values in input_list:
            for value in values:
                structure += [
                        (key, [
                            (value.__name__, [
                                ('object', value),
                                ('params', [
                                    ('None', {}) ] ) ] ) ] ) ]
        return structure

    def set_structure(self, input_list):
        structure = self._struct_transform(input_list)
        # delete old self before init !!!
        super(Steps, self).__init__(structure)

    def __iadd__(self, other):
        struc_other = self._struct_transform(other)
        return super(Steps, self).__iadd__(struc_other)

    def __sub__(self, other):
        pass

class CV(Dict):
    '''
    Make comfortable interface for work with cross-validation.
    '''
    def __init__(self, cv_list):
        super(CV, self).__init__(cv_list)

class Config:
    def __init__(self,  steps   = Steps([]), 
                        eval_cv = CV([]), 
                        scoring      = [], 
                        banned_steps = []):

        assert  type(steps)   == Steps  and \
                type(eval_cv) == CV     and \
                type(scoring) == list   and \
                type(banned_steps) == list

        self.steps   = steps
        self.eval_cv = eval_cv
        self.scoring = scoring
        self.banned_steps = banned_steps

    def dump(self, path):
        with open(path, 'wb') as f:
            for attr in sorted(self.__dict__.keys()):
                dump(getattr(self, attr), f)

    def load(self, path):
        with open(path, 'rb') as f:
            for attr in sorted(self.__dict__.keys()):
                setattr(self, attr, load(f))

class Results:
    def __init__(self):
        pass

class Pipeliner:
    def __init__(self, config):
        
        columns, transformers = zip(*config.steps.items())
        plan = DataFrame(columns = columns)

        def AddParams(key, params):
            for obj_params in product(*[[key], list(params.keys())]):
                yield '__'.join(obj_params)

        keys = []
        for transformer in transformers:
            keys += [[ value 
                    for key in transformer.keys()
                        for value in AddParams(key, transformer[key].params)]]
       
        def consist(steps, elements):
            pass


        for elements in product(*keys):
            print(config.banned_steps)
            for ban in config.banned_steps:
                print('ban')
                check = [step in elements for step in ban]
                print('ban = ', ban)
                print('elements = ', elements)
                print(check)
                print('')
            line = {}
            for column, value in zip(columns, elements):
                line[column] = [value]
            plan = plan.append(DataFrame.from_dict(line))

        self.plan = plan.reset_index(drop = True)[list(columns)]
        self.cfg = config
        self.best_params = {}
        self.scores = {}


    def fit_pipeline(self, X, y, line, scoring):
        key = ''.join(line.values) + scoring
        column = line.index[-1]
        classifier_key, params_key = line[column].split('__')
        params = self.cfg.steps[ column ][ classifier_key ].params[ params_key ]

        def create_object(line, column, config, scoring):
            full_key = line[column].split('__')
            object_key, params_key = full_key[0], full_key[1]
            params = config[object_key].params[params_key]
            if config[object_key].object == GridSearchCV:
                params['scoring'] = scoring
            return config[object_key].object( **params )

        steps = [ ( line[column].split('__')[0],
                    create_object(line, column, self.cfg.steps[ column ], scoring) )
                        for column in line.index ]
        pipeline = Pipeline(steps).fit(X, y)
        if type(pipeline.named_steps[classifier_key]) == GridSearchCV:
            self.best_params[key] = pipeline.named_steps[classifier_key].best_params_
        else:
            self.best_params[key] = {}

        return pipeline

    def score_pipeline(self, X, y, line, pipeline, scoring, eval_cv_key):
        key = ''.join(line.values) + scoring
        column = line.index[-1]
        classifier_key, params_key = line[column].split('__')
        params = self.cfg.steps[ column ][ classifier_key ].params[ params_key ]
    
        def create_object(line, column, config, scoring):
            key = ''.join(line.values) + scoring
            full_key = line[column].split('__')
            object_key, params_key = full_key
            params = config[object_key].params[params_key]
            if config[object_key].object == GridSearchCV:
                return params['estimator'].set_params( **self.best_params[key] )
            return config[object_key].object( **params )

        steps = [ ( line[column].split('__')[0],
                    create_object(line, column, self.cfg.steps[ column ], scoring) )
                        for column in line.index ]

        eval_cv_params = self.cfg.eval_cv[ eval_cv_key ].params
        eval_cv_object = self.cfg.eval_cv[ eval_cv_key ].object( **eval_cv_params )

        self.scores[key] = cross_val_score(  Pipeline(steps), X, y,
                                            scoring = scoring,
                                            cv      = eval_cv_object,
                                            n_jobs  = -1 )
        return self.scores[key]

    def get_features(self, X, line):
        return X

    def get_results(self, X, y, eval_cv_key,
                        scoring = 'accuracy',
                        featuring_steps = [],
                        results_file='temp.csv'):
        assert type(scoring) == str or type(scoring) == list
        assert all([x in self.plan.columns for x in featuring_steps])
        if type(scoring) == str:
            scoring = [scoring]

        without_featuring = [step for step in self.plan.columns
                                if step not in featuring_steps]

        grid_steps = dict()
        eval_steps = dict()
        columns = list(self.plan.columns)
        for metric in scoring:
            grid_steps[metric] = ['grid_' + metric + '_mean',
                                  'grid_' + metric + '_std',
                                  'grid_' + metric + '_best_params']
            eval_steps[metric] = ['eval_' + metric + '_mean',
                                  'eval_' + metric + '_std',
                                  'eval_' + metric + '_scores']
            columns += grid_steps[metric] + eval_steps[metric]

        ans = DataFrame(columns=columns, index=self.plan.index)
        if results_file != None:
            DataFrame(columns=columns).to_csv(results_file)
        ans[list(self.plan.columns)] = self.plan
        for index in self.plan.index:
            X_featured = self.get_features(X, self.plan.loc[index][featuring_steps])
            for metric in scoring:
                pipeline = self.fit_pipeline(   X_featured, y,
                                                self.plan.loc[index][without_featuring],
                                                scoring=metric)

                classifier_key = self.plan.loc[index][self.plan.columns[-1]].split('__')[0]
                clf = pipeline.named_steps[classifier_key]
                if type(clf) == GridSearchCV:
                    for score in clf.grid_scores_:
                        if score[0] == clf.best_params_:
                            ans.loc[index]['grid_' + metric + '_mean'] = score[1]
                            ans.loc[index]['grid_' + metric + '_std' ] = score[2].std()
                    ans.loc[index]['grid_' + metric + '_best_params']  = clf.best_params_

                scores = self.score_pipeline(X, y,
                                                self.plan.loc[index][without_featuring],
                                                pipeline,
                                                metric,
                                                eval_cv_key)
                ans.loc[index]['eval_' + metric + '_mean'] = mean(scores)
                ans.loc[index]['eval_' + metric + '_std'] = std(scores)
                ans.loc[index]['eval_' + metric + '_scores'] = str(scores)
            ans.loc[[index]].to_csv(results_file, header=False, mode='a')
        return ans
