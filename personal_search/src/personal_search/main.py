#!/usr/bin/env python
import sys
from personal_search.crew import PersonalSearchCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'O que a falta aos pregadores de hoje?'
    }
    PersonalSearchCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        PersonalSearchCrew().crew().train(n_iterations=int(sys.argv[1]))

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
