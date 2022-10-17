from unittest import TestCase
from Agents.Agent import Agent
from usr_func.set_resume_state import set_resume_state


class TestAgent(TestCase):

    def setUp(self) -> None:
        set_resume_state(True)
        self.ag = Agent()

    def test_agent_run(self):
        self.ag.run()



