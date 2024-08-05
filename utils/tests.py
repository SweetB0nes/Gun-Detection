import math
import numbers
import sys
import unittest

import numpy


def run_tests(recieved, expected, inputs, test_info=''):
    
    # Checking the data types
    condition = type(recieved) == type(expected)
    message = (test_info +
        '\n\nERROR: Type mismatch!\n' +
        'Recieved type {}\n'.format(type(recieved)) +
        'Expected type {}\n'.format(type(expected)) +
        'Fix the type and resubmit')
    
    assert condition, message

    # Checking dictionary
    if isinstance(expected, dict):
        # Checking dictionary keys
        message = (
            test_info + '\n' +
            'Keys in expected result and in the answer do not coincide:\n' +
            'Expected keys: ' + str(set(expected.keys())) + '\n' +
            'Recieved keys: ' + str(set(recieved.keys())) + '\n' +
            'Extra keys in the answer: ' + 
            str(set(recieved.keys()).difference(set(expected.keys()))) +
            'Missing keys in the answer: ' + 
            str(set(expected.keys()).difference(set(recieved.keys()))))

        condition = (
            len(set(expected.keys()).difference(set(recieved.keys()))) == 0 and
            len(set(recieved.keys()).difference(set(expected.keys()))) == 0)
        
        assert condition, message

        # Checking dictionary entries
        for key in expected:
            run_tests(
                recieved[key],
                expected[key],
                inputs,
                test_info=test_info + '->' + str(key)
            )

    # Checking list
    elif isinstance(expected, list):
        # Checking list size
        condition = len(recieved) == len(expected)
        message = (test_info + '\n' +
            'Expected result and the answer have different number of items:\n' +
            'Expected len: {}\n'.format(len(expected)) +
            'Recieved len: {}\n'.format(len(recieved)))
        
        assert(condition, message)

        # Checking list entries
        for index in range(len(recieved)):
            self.get_tests(
                recieved[index],
                expected[index],
                inputs,
                test_info=test_info + '->' + str(index)
            )

    # Checking number
    elif isinstance(expected, numbers.Number):
        condition = (math.fabs(expected - recieved) < 1.0e-8)
        message = (test_info + '\n' +
            'Expected result and the answer are numerically different' + '\n' +
            'Expected value: ' + str(expected) + '\n' +
            'Recieved value: ' + str(recieved))
        assert condition, message

    # Checking numpy array
    elif isinstance(expected, numpy.ndarray):
        # Checking numpy dtype
        condition = (recieved.dtype == expected.dtype)
        message = (test_info + '\n' +
                'Numpy array type mismatch:\n' +
                'Expected: {}'.format(expected.dtype) + '\n' +
                'Recieved: {}'.format(recieved.dtype))
        assert condition, message

        # Checking numpy singleton
        if len(expected.shape) == 0:
            # Checking the second array also holds a singleton
            condition = (len(recieved.shape) == 0)
            message = (test_info + '\n' +
                    'Expected scalar\n' +
                    'Got tensor')
            assert condition, message

        # Checking numpy array   
        else:
            # Checking numpy shape
            condition = numpy.abs(
                        numpy.array(recieved.shape) -
                        numpy.array(expected.shape)
                    ).mean() < 1.0e-8
            
            message = (test_info + '\n' +
                    'Shape mismatch:\n' +
                    'Expected: {}\n'.format(str(expected.shape)) +
                    'Recieved: {}'.format(str(recieved.shape)))
            
            assert condition, message

        # Checking exact data types
        if recieved.dtype in {numpy.bool_, numpy.byte, numpy.ubyte}:
            # Checking equality of all items in arrays
            condition = (recieved == expected).all()
            message = (test_info + '\n' +
                    'Values mismatch:\n' +
                    'Expected: {}\n'.format(str(expected)) +
                    'Recieved: {}'.format(str(recieved)))
            assert condition, message

        # Checking float data types
        else:
            # Checking deviations of elements
            condition = numpy.abs(recieved - expected).mean() < 1.0e-8
            message = (test_info + '\n' +
                'Values mismatch:\n' +
                'Expected: {}\n'.format(str(expected)) +
                'Recieved: {}'.format(str(recieved)))
            assert condition, message

    elif isinstance(expected, torch.tensor):
        run_tests(recieved.detach().numpy(), 
                  expected.detach().numpy(), 
                  inputs,
                  test_info)
