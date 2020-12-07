

def embed_agent(multienv):
    policy = load_policy()
    env = CurryVecEnv(multienv, policy)
