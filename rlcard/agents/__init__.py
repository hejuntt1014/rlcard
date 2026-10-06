from importlib.util import find_spec

if find_spec('torch') is not None:
    from rlcard.agents.dqn_agent import DQNAgent as DQNAgent
    from rlcard.agents.nfsp_agent import NFSPAgent as NFSPAgent

from rlcard.agents.cfr_agent import CFRAgent
from rlcard.agents.human_agents.limit_holdem_human_agent import HumanAgent as LimitholdemHumanAgent
from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent as NolimitholdemHumanAgent
from rlcard.agents.human_agents.leduc_holdem_human_agent import HumanAgent as LeducholdemHumanAgent
from rlcard.agents.human_agents.blackjack_human_agent import HumanAgent as BlackjackHumanAgent
from rlcard.agents.human_agents.uno_human_agent import HumanAgent as UnoHumanAgent
from rlcard.agents.random_agent import RandomAgent
