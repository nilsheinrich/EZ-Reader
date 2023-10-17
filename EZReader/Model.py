# This is the core EZ Reader class. Its internal states instantiate the model's components and generates eye movements.


def run_EZReader(parameters, corpus, NRuns=1, sentences="all", timeout=20, verbose=False):

    if verbose:
        print("to run the E-Z Reader model type for example:\n"
              "run_EZReader(C:/Users/NilsWendel/Documents/GitLab/ez-reader/EZ/Data/paramteters.txt,\n"
              "C:/Users/NilsWendel/Documents/GitLab/ez-reader/EZ/Data/SRC98Corpus.txt,\n"
              "20,\n"
              "timeout=300)\n"
              "\n"
              "This will start the model with the specified parameters and corpus file.\n"
              "It will simulate 20 complete readings of the specified corpus.\n"
              "The model will stop running after 5 min (300s) regardless of whether it's done or not."
              )
    import random as r
    import numpy as np
    import pandas as pd

    import time

    # from this directory
    from EZReader import Display
    from EZReader import Fixation
    from EZReader import Process
    from EZReader import Saccade

    # # Lexical Processing:
    # parameters["Alpha1"]
    # parameters["Alpha2"]
    # parameters["Alpha3"]
    # parameters["Delta"]

    # # Post-Lexical Processing:
    # parameters["I"]
    # parameters["pF"]
    # parameters["ITarget"]  # unused?
    # parameters["pFTarget"]

    # # Saccadic Latency:
    # parameters["M1"]
    # parameters["M2"]
    # parameters["S"]
    # parameters["Xi"]

    # # Saccadic Error:
    # parameters["Psi"]
    # parameters["Omega1"]
    # parameters["Omega2"]
    # parameters["Eta1"]
    # parameters["Eta2"]

    # # Vision & misc.:
    # parameters["V"]
    # parameters["Epsilon"]
    # parameters["A"]
    # parameters["Lambda"]
    # parameters["SigmaGamma"]

    #####################################################################

    # trace of fixations
    fixations = []   # create empty list for trace of fixations
    # the above dic will "collect" every fixation over all runs/subjects

    # timeout: how much time until timeout (in seconds)
    time_stamp = time.time()  # begin counting from here

    # Beginning of run loop:
    for run in range(NRuns):

        if time.time() > time_stamp + timeout:
            #print("break run loop")
            break

        # Beginning of sentence loop:
        for sn_nr, sentence in enumerate(corpus.sentences, 1):
            if (type(sentences) == str and sentences != "all") or (type(sentences) == int and sn_nr != sentences) or (type(sentences) == list and sn_nr not in sentences):
                continue

            if time.time() > time_stamp + timeout:
                #print("break sentence loop")
                break

            number_of_words = len(sentence.words)

            # Initialize fixation:
            fixationN = 1
            trace_array = []  # open array with trace of fixations
            f = Fixation.Fixation(number=fixationN, pos=sentence.word(1).OVP, word=1)

            # Start L1:
            activeProcesses = []  # open array with active processes
            WN = 1
            p = Process.initializeL1(parameters, sentence.word(WN), True)
            rate = p.calcRate(parameters, sentence.word(WN), f.pos)
            p.dur = p.dur * rate
            activeProcesses.append(p)

            # Initialize integration-failure flags for all words:
            integrationFailure = [False] * number_of_words

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Model starts reading single sentence
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sentenceDone = False
            while not sentenceDone:

                # Identify processing w/ shortest duration:
                #completeProcess = min(activeProcesses, key=lambda activeProcess: activeProcess.dur)
                #activeProcesses.remove(completeProcess)
                #print(completeProcess.dur)
                #print(f"list: {activeProcesses}, list_after_pop: {activeProcesses.remove(completeProcess)}")

                minProcessDuration = activeProcesses[0].dur
                minProcessID = 0
                for i, activeProcess in enumerate(activeProcesses):
                    if activeProcess.dur < minProcessDuration:
                        minProcessDuration = activeProcess.dur
                        minProcessID = i

                # Store attributes of shortest process:
                completeProcess = activeProcesses.pop(minProcessID)

                # Decrement all remaining process durations:
                for activeProcess in activeProcesses:
                    activeProcess.dur -= completeProcess.dur

                # Increase fixation duration (except if saccade ongoing):
                if completeProcess.name != "S":
                    f.dur += completeProcess.dur

                # **********************************************************
                # DETERMINE & EXECUTE NEXT MODEL STATE
                # **********************************************************

                # PRE-ATTENTIVE VISUAL PROCESSING (V):
                if completeProcess.name == "V":
                    # Adjust L1 rate:
                    for activeProcess in activeProcesses:
                        if activeProcess.name == "L1":
                            activeProcess.dur /= rate
                            rate = p.calcRate(parameters, sentence.word(WN), f.pos)
                            activeProcess.dur *= rate

                # FAMILIARITY CHECK (L1):
                elif completeProcess.name == "L1":

                    # Determine if integration is on-going for previous word:
                    ongoingI = False
                    for activeProcess in activeProcesses:
                        if activeProcess.name == "I" and activeProcess.wn == (WN - 1):
                            ongoingI = True

                    # Start L2:
                    p = Process.initializeL2(parameters, sentence.word(WN), ongoingI)
                    activeProcesses.append(p)

                    # Cancel any pending M1 & start M1:
                    if WN < number_of_words:
                        p = Process.initializeM1(parameters, activeProcesses, f.pos, sentence.word(WN + 1).OVP, WN + 1)
                        activeProcesses.append(p)

                # LEXICAL ACCESS (L2):
                elif completeProcess.name == "L2":

                    # Determine if integration of the previous word failed:
                    integrationFail = False
                    integrationFail_wn = 0  # integration failure word

                    for activeProcess in reversed(activeProcesses):
                        if activeProcess.name == "I":
                            integrationFail_wn = activeProcess.wn
                            del activeProcess

                    # Prohibit integration failure from potentially happening twice on a word.
                    if integrationFail_wn > 0 and not integrationFailure[integrationFail_wn - 1]:  # -1 since array index starts at 0
                        integrationFailure[integrationFail_wn - 1] = True
                        integrationFail = True
                        WN = integrationFail_wn

                    # Slow integration failure:
                    if integrationFail:

                        # Note: A and/or L1 do NOT have to be removed from active processes because L2 just finished
                        # (i.e., A and L1 cannot be ongoing).

                        # Start A:
                        if WN < number_of_words:
                            p = Process.initializeA(parameters, WN)
                            activeProcesses.append(p)

                        # Cancel any pending M1 & start M1:
                            p = Process.initializeM1(parameters, activeProcesses, f.pos, sentence.word(WN).OVP, WN)
                            activeProcesses.append(p)

                    # Integration was successful:
                    else:

                        # Start I:
                        p = Process.initializeI(parameters, WN)  # integration of current word begins
                        activeProcesses.append(p)

                        # Start A:
                        if WN < number_of_words:
                            p = Process.initializeA(parameters, completeProcess.wn + 1)  # Attention is directed
                            # towards next word
                            activeProcesses.append(p)

                # POST-LEXICAL INTEGRATION (I):
                elif completeProcess.name == "I":

                    # Rapid integration failure:
                    PrintegrationFailure = r.random()
                    if not integrationFailure[completeProcess.wn - 1] and WN > 1 and (PrintegrationFailure < parameters["pF"]):

                        # Flag word for integration failure:
                        integrationFailure[completeProcess.wn - 1] = True

                        # Stop A, L1, and/or L2:
                        for activeProcess in reversed(activeProcesses):
                            if activeProcess.name in ["A", "L1", "L2"]:
                                del activeProcess

                        # Start A:
                        if WN < number_of_words:
                            WN = completeProcess.wn  # i.e., shift attention to source of integration difficulty
                            p = Process.initializeA(parameters, WN)
                            activeProcesses.append(p)

                            # Cancel any pending M1 & start M1:
                            p = Process.initializeM1(parameters, activeProcesses, f.pos, sentence.word(WN).OVP, WN)
                            activeProcesses.append(p)

                    else:
                        if completeProcess.wn == number_of_words:
                            sentenceDone = True

                # ATTENTION SHIFT (A):
                elif completeProcess.name == "A":

                    # Shift attention to word:
                    WN = completeProcess.wn

                    # Determine if integration is on-going for previous word:
                    ongoingI = False
                    for activeProcess in activeProcesses:
                        if activeProcess.name == "[I]" and activeProcess.wn == WN - 1:
                            ongoingI = True

                    # Start L1:
                    print(f"ongoingI: {ongoingI}; word:{sentence.word(WN)}; word_number: {WN}; fixation_location_word: {f.word}")
                    p = Process.initializeL1(parameters, sentence.word(WN), ongoingI)
                    rate = p.calcRate(parameters, sentence.word(WN), f.pos)
                    p.dur *= rate
                    activeProcesses.append(p)

                # LABILE SACCADIC PROGRAMMING (M1):
                elif completeProcess.name == "M1":

                    # Start M2:
                    s = Saccade.Saccade(completeProcess.IntendedSaccadeLength, f.dur)
                    p = Process.initializeM2(parameters, completeProcess.wn, s)
                    activeProcesses.append(p)

                # NON-LABILE SACCADIC PROGRAMMING (M2):
                elif completeProcess.name == "M2":

                    # Start S:
                    p = Process.initializeS(parameters, completeProcess.ActualSaccadeLength, completeProcess.wn)
                    activeProcesses.append(p)

                # SACCADE EXECUTION (S):
                elif completeProcess.name == "S":

                    # Terminate previous fixation:
                    trace_array.append(f)

                    # Start new fixation:
                    newPosition = trace_array[fixationN - 1].pos + completeProcess.ActualSaccadeLength  # -1 since we want the previous fixation and the array starts at index 0
                    fixationN += 1
                    fixatedWord = None  # introduce variable to avoid referencing before assignment
                    if newPosition < 0:
                        newPosition = 0
                    lastChar = sentence.word("last").posN
                    if newPosition > lastChar:
                        newPosition = lastChar
                    for i, word in enumerate(sentence.words, 1):
                        if word.pos0 <= newPosition < word.posN:
                            fixatedWord = i
                            break

                    f = Fixation.Fixation(number=fixationN, pos=newPosition, word=fixatedWord)

                    # Start V:
                    p = Process.initializeV(parameters, WN)
                    activeProcesses.append(p)

                    # Start M1 (automatic re-fixation):
                    PrRefixate = r.random()
                    saccadeError = np.abs(sentence.word(completeProcess.wn).OVP - f.pos)
                    if PrRefixate < (parameters["Lambda"] * saccadeError):
                        p = Process.initializeM1(parameters, activeProcesses, f.pos, sentence.word(completeProcess.wn).OVP, completeProcess.wn)
                        activeProcesses.append(p)

                    # **********************************************************
                    # MODEL STATE EXECUTED
                    # **********************************************************

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # MODEL HAS FINISHED READING SENTENCE
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            activeProcesses.clear()  # remove any remaining active processes

            Display.traceList(fixations, sentence, sn_nr, trace_array, run + 1)

            trace_array.clear()  # remove trace of fixations

            # End of sentences loop

        # End of run loop

    results = pd.DataFrame(fixations, columns=["fixationN", "duration", "position", "wordN", "freq", "freqClass", "sentenceN", "runN"])

    if time.time() > time_stamp + timeout:
        output_file = open("issues_output.txt", "w")
        print(f"---timeout--- parameter values:{parameters}", file=output_file)
        print("time exceeded")
    return results
