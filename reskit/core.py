from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from collections import OrderedDict
from itertools import product
from pickle import dump, load
from pprint import pprint, pformat
from pandas import DataFrame, MultiIndex
from numpy import mean, std
from time import time

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
        super(Steps, self).__init__(structure)

    def __iadd__(self, other):
        struc_other = self._struct_transform(other)
        return super(Steps, self).__iadd__(struc_other)

    def __sub__(self, other):
        pass

class Config:
    def __init__(self,  steps   = Steps([]), 
                        eval_cv =       None, 
                        scoring      =  [], 
                        banned_steps =  []):

        assert  type(steps)   == Steps  and \
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

        for elements in product(*keys):
            line = {}
            for column, value in zip(columns, elements):
                line[column] = [value]
            plan = plan.append(DataFrame.from_dict(line))

        self.plan = plan.reset_index(drop = True)[list(columns)]
        self.cfg  = config
        self.best_params = dict()
        self.scores      = dict()
        self.temp_X      = dict()

    def _create_object(self, line, column, config, scoring):
        full_key = line[column].split('__')
        object_key, params_key = full_key
        params = config[object_key].params[params_key]
        if config[object_key].object == GridSearchCV:
            params['scoring'] = scoring
        return config[object_key].object( **params )

    def get_features(self, X, line):
        if len(line) == 0:
            return X
        columns = iter(line.index)
        prev_column = 'init'

        if 'init' in self.temp_X:
            for column in columns:
                if self.featuring_steps[column] != line[column]:
                    break
                else:
                    prev_column = column
        else:
            column = next(columns)
            self.temp_X['init'] = X
            
        self.featuring_steps = line
        last_key = line.index[-1]

        if prev_column == last_key:
            return self.temp_X[last_key]

        for column in [column] + list(columns):
            key_o, key_p = line[column].split('__')
            key_object = [
                    column,
                    key_o,
                    'object' ]
            key_params = [
                    column,
                    key_o,
                    'params',
                    key_p ]
            obj     = self.cfg.steps.get(*key_object)
            params  = self.cfg.steps.get(*key_params)
            self.temp_X[column] = obj( **params ).fit_transform(
                    self.temp_X[prev_column] )
            prev_column = column

        return self.temp_X[last_key]



        return column

    def get_pipeline(self, line, scoring):

        steps = [ ( line[column].split('__')[0],
                    self._create_object(line, column, self.cfg.steps[ column ], scoring) )
                        for column in line.index ]

        return Pipeline(steps)

    def get_scores(self, X, y, line, classifier, scoring):

        steps = [ ( line[column].split('__')[0],
                    self._create_object(line, column, self.cfg.steps[ column ], scoring) )
                        for column in line.index[:-1] ]
        steps += [(line.index[-1], classifier)]
        
        scores = cross_val_score(   Pipeline(steps), X, y,
                                    scoring = scoring,
                                    cv      = self.cfg.eval_cv,
                                    n_jobs  = -1 )
         
        return scores

    def get_results(self, X, y,
                        scoring = 'accuracy',
                        featuring_steps = [],
                        results_file='temp.csv'):
        assert type(scoring) == str or type(scoring) == list
        assert all(self.plan.columns[:len(featuring_steps)] == featuring_steps)
        if type(scoring) == str:
            scoring = [scoring]

        columns = list(self.plan.columns)
        without_featuring = [step for step in columns
                                if step not in featuring_steps]

        for metric in scoring:
            grid_steps = [  'grid_' + metric + '_mean',
                            'grid_' + metric + '_std',
                            'grid_' + metric + '_best_params']

            eval_steps = [  'eval_' + metric + '_mean',
                            'eval_' + metric + '_std',
                            'eval_' + metric + '_scores']

            columns += grid_steps + eval_steps

        ans = DataFrame(columns=columns, index=self.plan.index)

        if results_file != None:
            DataFrame(columns=columns).to_csv(results_file)

        ans[list(self.plan.columns)] = self.plan

        N = len(self.plan.index)
        for index in self.plan.index:
            line = self.plan.loc[index]
            print('\nLine: {} / {} \n'.format(index + 1, N), line, '\n')

            start = time()
            X_featured = self.get_features(X, line[featuring_steps])
            print('\n\tFeaturing: ', time() - start, ' sec\n')

            for metric in scoring:
                pipeline = self.get_pipeline(   line[without_featuring],
                                                scoring=metric)
                start = time()
                pipeline.fit(X_featured, y)
                print('\nPipeline: ', pipeline)
                print('\n\tGridSearching: ', time() - start, ' sec\n\n')

                classifier_key = line[self.plan.columns[-1]].split('__')[0]
                clf = pipeline.named_steps[classifier_key]

                key = ''.join(line.values) + metric
                if type(clf) == GridSearchCV:
                    self.best_params[key] = clf.best_params_

                    for i, params in enumerate(clf.cv_results_['params']):
                        if params == self.best_params[key]:

                            ans.loc[index]['grid_' + metric + '_mean'] = \
                                clf.cv_results_['mean_test_score'][i]

                            ans.loc[index]['grid_' + metric + '_std'] =  \
                                clf.cv_results_['std_test_score'][i]

                            ans.loc[index]['grid_' + metric + '_best_params'] = \
                                str(params)

                            print(metric + '_grid_mean = ', clf.cv_results_['mean_test_score'][i])
                            print(metric + '_grid_std = ',  clf.cv_results_['std_test_score'][ i])
                            print(metric + '_grid_best_params: ', params, '\n\n')
                    clf = clf.best_estimator_
                else:
                    print('Vanilla ', clf, '\n\n')
                    self.best_params[key] = {}

                start = time()
                scores = self.get_scores(   X_featured, y,
                                            line[without_featuring],
                                            clf,
                                            metric )
                print('\tScoring_' + metric + ': ', time() - start, ' sec\n\n')

                scores_mean = mean(scores)
                scores_std  = std(scores)

                ans.loc[index]['eval_' + metric + '_mean']  = scores_mean
                ans.loc[index]['eval_' + metric + '_std']   = scores_std
                ans.loc[index]['eval_' + metric + '_scores']= str(scores)
                print(metric + '_eval_mean ', scores_mean)
                print(metric + '_eval_std ' , scores_std)
                print(metric + '_eval_scores',str(scores))
                self.scores[key] = scores

            ans.loc[[index]].to_csv(results_file, header=False, mode='a')
        return ans
