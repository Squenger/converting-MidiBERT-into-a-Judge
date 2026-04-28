import random
import math

def corrupt_tokens(tokens, tokenizer, severity=0.5):
    """
    corrupts a piece of music in a realistic way.
    - severity : from 0.0 (perfect, 10/10) to 1.0. 
    """
    corrupted_tokens = []
    
    # we limit the probability of corruption to 40% for 1 severity
    MAX_CORRUPTION_RATE = 0.40
    prob_alteration = severity * MAX_CORRUPTION_RATE
    
    # the target score is indexed on the severity
    target_score = 10.0 * (1.0 - math.sqrt(severity)) 
    # we use sqrt to make the score decrease faster at the beginning 
    # because one false note in a perfect melody ruins the whole piece

    family_of_alteration = ["pitch","rhythm","duration","velocity"]
    
    for token_id in tokens:
        # get the event string from the tokenizer to check its type (pitch, position, etc.)
        event_str = tokenizer[token_id]
        
        # choose whether the token is corrupted
        if random.random() > prob_alteration:
            corrupted_tokens.append(token_id)
            continue

        nb_type_of_alteration=random.randint(1,len(family_of_alteration))
        active_alteration = random.sample(family_of_alteration,nb_type_of_alteration)


            
            # pitch error
        if "pitch" in event_str.lower() and "pitch" in active_alteration:
            type_erreur = random.choice(["semi_tone", "octave", "supp"])
            if type_erreur == "semi_tone":
                corrupted_tokens.append(token_id + random.choice([-2, -1, 1, 2]))
            elif type_erreur == "octave":
                corrupted_tokens.append(token_id + random.choice([-12, 12]))
            # if "supp", don't add it to the list 
        
        # rhythm error (position/time)
        elif ("position" in event_str.lower() or "bar" in event_str.lower()) and "rhythm" in active_alteration:
            type_erreur = random.choice(["silence", "compression"])
            if type_erreur == "silence":
                # creates a silence gap
                corrupted_tokens.extend([token_id, token_id]) 
            # if "compression" we ignore it
            
        # duration error
        elif "duration" in event_str.lower() and "duration" in active_alteration:
            # divide the duration by 2 (staccato)

            corrupted_tokens.append(max(0, token_id - 4)) 

        # velocity error
        elif "velocity" in event_str.lower() and "velocity" in active_alteration:
            # force the note to be played very loud

            corrupted_tokens.append(token_id + 10) 
            
        else:
            corrupted_tokens.append(token_id)
            
    return corrupted_tokens, target_score