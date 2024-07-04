#!/usr/bin/env python
import sys
from personal_search.crew import PersonalSearchCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'O mais importante é que vocês vivam em sua comunidade de maneira digna das boas-novas de Cristo. Então, quando eu for vê-los novamente, ou mesmo quando ouvir a seu respeito, saberei que estão firmes e unidos em um só espírito e em um só propósito, lutando juntos pela fé que é proclamada nas boas-novas.'
    }
    PersonalSearchCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'topic': 'O mais importante é que vocês vivam em sua comunidade de maneira digna das boas-novas de Cristo. Então, quando eu for vê-los novamente, ou mesmo quando ouvir a seu respeito, saberei que estão firmes e unidos em um só espírito e em um só propósito, lutando juntos pela fé que é proclamada nas boas-novas.'
    }
    try:
        PersonalSearchCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
