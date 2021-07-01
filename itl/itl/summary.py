from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import numpy as np
import tensorflow as tf

from tensorflow.python.client import timeline

__all__ = ['Summary']


class Summary(object):

    def __init__(self, session, params):
        self.session = session
        self.params = params

    def construct(self, subfolder_path):
        if self.params['path'] != '':
            if self.params['step'] >= 0 and self.params['step'] < np.inf:
                self.merged_summaries_ = tf.summary.merge_all()
            else:
                self.merged_summaries_ = []
            if self.params['graph'] >= 0 and self.params['graph'] < np.inf:
                self.summary_writer_ = tf.summary.FileWriter(
                    self.params['path'] + '/' + subfolder_path,
                    self.session.graph)
                self.run_options_ = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                self.run_metadata_ = tf.RunMetadata()
            else:
                self.summary_writer_ = tf.summary.FileWriter(
                    self.params['path'] + '/' + subfolder_path)
                self.run_options_ = tf.RunOptions(
                    trace_level=tf.RunOptions.NO_TRACE)
                self.run_metadata_ = None
        else:
            self.merged_summaries_ = []
            self.summary_writer_ = None
            self.run_options_ = None
            self.run_metadata_ = None
        self.reset()
        return self

    def allow_dump_metadata(self):
        if (self.params['graph'] <= 0 or self.params['graph'] >= np.inf or
           self.summary_writer_ is None):
            return False
        return ((self.iter % self.params['graph'] == 0) or
                (self.time > self.params['time']))

    def dump_metadata(self):
        if self.run_metadata is not None and self.run_options is not None:
            tag = 'step_{}'.format(self.iter)
            self.summary_writer.add_run_metadata(self.run_metadata, tag)
            self.chrome_trace(tag)
        return self

    def allow_dump_summaries(self):
        if (self.params['step'] <= 0 or self.params['step'] >= np.inf or
           self.summary_writer_ is None):
            return False
        return ((self.iter % self.params['step'] == 0) or
                (self.time > self.params['time']))

    def dump_summaries(self, summaries):
        self.summary_writer.add_summary(summaries, self.iter)
        return self

    def dump(self, summaries):
        if self.summary_writer is not None:
            if self.params['step'] > 0 and self.params['step'] < np.inf:
                if len(summaries) > 0:
                    self.dump_summaries(*summaries)
            if self.params['graph'] > 0 and self.params['graph'] < np.inf:
                self.dump_metadata()
        return self

    def __call__(self, fetches=None):
        if self.allow_dump_summaries():
            self.dump_summaries(fetches)
        if self.allow_dump_metadata():
            self.dump_metadata()
        if self.allow_dump_metadata() or self.allow_dump_summaries():
            self.reset_timer()
        self.iter_inc()
        return self

    @property
    def iter(self):
        return self.i_

    def iter_inc(self):
        self.i_ = self.i_ + 1
        return self

    @property
    def summary_writer(self):
        return self.summary_writer_

    @property
    def merged_summaries(self):
        if self.allow_dump_summaries():
            return [self.merged_summaries_]
        else:
            return []

    @property
    def run_options(self):
        if self.allow_dump_metadata():
            return self.run_options_
        else:
            return None

    @property
    def run_metadata(self):
        if self.allow_dump_metadata():
            return self.run_metadata_
        else:
            return None

    @property
    def time(self):
        return time.time() - self.init_time_

    def reset(self):
        self.i_ = 0
        self.reset_timer()
        return self

    def reset_timer(self):
        self.init_time_ = time.time()
        return self

    def chrome_trace(self, name='timeline'):
        if self.params['path'] == '':
            return self
        fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(self.params['path'] + '/' + name + '.json', 'w') as fd:
            fd.write(chrome_trace)
        return self
