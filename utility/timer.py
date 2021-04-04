from time import strftime, gmtime, time
from collections import defaultdict
import tensorflow as tf

from utility.aggregator import Aggregator
from utility.display import pwc


def timeit(func, *args, name=None, to_print=False, **kwargs):
	start_time = gmtime()
	start = time()
	result = func(*args, **kwargs)
	end = time()
	end_time = gmtime()

	if to_print:
		pwc(f'{name if name else func.__name__}: '
            f'Start "{strftime("%d %b %H:%M:%S", start_time)}"', 
            f'End "{strftime("%d %b %H:%M:%S", end_time)}"' 
            f'Duration "{end - start:.3g}s"', color='blue')

	return end - start, result

class Timer:
    aggregators = defaultdict(Aggregator)

    def __init__(self, summary_name, period=None, mode='average', to_log=True):
        self._to_log = to_log
        if self._to_log:
            self._summary_name = summary_name
            self._period = period
            assert mode in ['average', 'sum']
            self._mode = mode

    def __enter__(self):
        if self._to_log:
            self._start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._to_log:
            duration = time() - self._start
            aggregator = self.aggregators[self._summary_name]
            aggregator.add(duration)
            if self._period is not None and aggregator.count >= self._period:
                if self._mode == 'average':
                    duration = aggregator.average()
                    duration = (f'{duration*1000:.3g}ms' if duration < 1e-1 
                                else f'{duration:.3g}s')
                    pwc(f'{self._summary_name} duration: "{duration}" averaged over {self._period} times', color='blue')
                    aggregator.reset()
                else:
                    duration = aggregator.sum
                    pwc(f'{self._summary_name} duration: "{duration}" for {aggregator.count} times', color='blue')

    def reset(self):
        aggregator = self.aggregators[self._summary_name]
        aggregator.reset()
    
    def average(self):
        return self.aggregators[self._summary_name].average()
        
    def last(self):
        return self.aggregators[self._summary_name].last
    
    def total(self):
        return self.aggregators[self._summary_name].total

class TBTimer:
    aggregators = defaultdict(Aggregator)

    def __init__(self, summary_name, period=1, to_log=True, print_terminal_info=False):
        self._to_log = to_log
        if self._to_log:
            self._summary_name = summary_name
            self._period = period
            self._print_terminal_info = print_terminal_info

    def __enter__(self):
        if self._to_log:
            self._start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._to_log:
            duration = time() - self._start
            aggregator = self.aggregators[self._summary_name]
            aggregator.add(duration)
            if aggregator.count >= self._period:
                duration = aggregator.average()
                step = tf.summary.experimental.get_step()
                tf.summary.scalar(f'timer/{self._summary_name}', duration, step=step)
                aggregator.reset()
                if self._print_terminal_info:
                    pwc(f'{self._summary_name} duration: "{duration}" averaged over {self._period} times', color='blue')


class LoggerTimer:
    def __init__(self, logger, summary_name, to_log=True):
        self._to_log = to_log
        if self._to_log:
            self._logger = logger
            self._summary_name = summary_name

    def __enter__(self):
        if self._to_log:
            self._start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._to_log:
            duration = time() - self._start
            self._logger.store(**{self._summary_name: duration})
