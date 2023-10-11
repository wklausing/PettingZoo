from SabreEnvironment_v3 import raw_env

env = raw_env()
env.reset()
stepCount = 0
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated:
        env.step(None)
        print(f'Agent {agent} terminated')

    elif truncated:
        env.step(None)
        print('Truncated')

    else:
        if 'cp' in agent:
            cpAgent = env.agentsDict[agent]
            cpAgent.money += reward
            if cpAgent.money <= 0:
                print('CP agent is out of money.')
                quit()

            elif all(cdn.soldContigent <= 0 for cdn in env.cdns): # Buy contigent from cdn with lowest price and steer client to it.
                indexOfCheapestCDN, _ = min(enumerate(env.cdns), key=lambda item: item[1].pricingFactor)
                buyContigent_action = indexOfCheapestCDN
                steerClient_action = [indexOfCheapestCDN]

            else: # Steer client to CDN with already bought contigent. Don't buy anything.
                buyContigent_action = env.cdnCount + 1
                cdn = next((cdn for cdn in env.cdns if cdn.soldContigent > 0), None)
                steerClient_action = [cdn.id[3:]]
            
            action = {
                'buyContigent': buyContigent_action,
                'steerClient': steerClient_action
            }

        elif 'cdn' in agent:
            cdnAgent = env.agentsDict[agent]

            if '_0' in agent:
                opponentsPrice = env.agentsDict['cdn_1'].pricingFactor
            else:
                opponentsPrice = env.agentsDict['cdn_0'].pricingFactor

            if cdnAgent.soldContigent > 0 and cdnAgent.pricingFactor < 5:
                cdnAgent.pricingFactor = round(opponentsPrice + 0.1, 3)
            elif cdnAgent.soldContigent <= 0 and cdnAgent.pricingFactor > 0:
                cdnAgent.pricingFactor = round(opponentsPrice - 0.1, 3)
            else:
                print(f'{agent} agent has a price of {cdnAgent.pricingFactor}.')
            
            action = {
                'changePrice': [cdnAgent.pricingFactor],
                'moveEdgeServer': 1
            }

        else:
            raise Exception(f'Agent {agent} in main function not found.')

        env.step(action)
    stepCount += 1
    if stepCount >= 10000: 
        print('Step count exceeded.')
        break
env.close()