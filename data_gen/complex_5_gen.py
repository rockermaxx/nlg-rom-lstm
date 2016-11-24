import random

import numpy as np


# Categories for each weather parameter
def _wind_cat(wind):
    if wind < 0.2:
        return 1
    elif wind < 0.5:
        return 2
    elif wind < 0.7:
        return 3
    else:
        return 4


def _prec_cat(prec):
    if prec < 0.1:
        return 1
    elif prec < 0.4:
        return 2
    elif prec < 0.7:
        return 3
    else:
        return 4


def _temp_cat(temp):
    if temp < 0.3:
        return 1
    elif temp < 0.5:
        return 2
    elif temp < 0.7:
        return 3
    else:
        return 4


def _humid_cat(humid):
    if humid < 0.4:
        return 1
    elif humid < 0.7:
        return 2
    else:
        return 3


def _wind_and_prec(wind, prec):
    w = _wind_cat(wind)
    p = _prec_cat(prec)

    if w == 1:
        if p == 1:
            if np.random.rand() < 0.5:
                return "without any traces of rain or heavy winds"
            else:
                return "without any wind or rain"
        elif p == 2:
            if np.random.rand() < 0.5:
                return "with mild rain showers"
            else:
                return ""
        elif p == 3:
            return "with thunderstorms"
        else:
            return "with thunderstorms and hailstones"
    elif w == 2:
        if p == 1:
            if np.random.rand() < 0.5:
                return "with light breeze and without any traces of rain"
            else:
                return "without any wind or rain"
        elif p == 2:
            return "with light breeze and mild rain showers"
        elif p == 3:
            return "with loud thunderstorms"
        else:
            return "with thunderstorms and hailstones"
    elif w == 3:
        if p == 1:
            if np.random.rand() < 0.5:
                return "with moderately strong winds and without any traces of rain"
            else:
                return "with moderately strong winds"
        elif p == 2:
            return "with moderately strong winds and mild rain showers"
        elif p == 3:
            return "with strong winds and loud thunderstorms"
        else:
            return "with strong winds, thunderstorms and hailstones"
    else:
        if p == 1:
            if np.random.rand() < 0.5:
                return "with cyclonic  winds and traces of rain"
            else:
                return "with cyclonic winds"
        elif p == 2:
            return "with cyclonic  winds and mild rain showers"
        elif p == 3:
            return "with cyclonic winds and loud thunderstorms"
        else:
            return "with cyclonic winds, thunderstorms and hailstones"


def _temp_and_humid(temp, humid):
    t = _temp_cat(temp)
    h = _wind_cat(humid)

    if t == 1:
        if h == 1:
            return "a very dry and cold day"
        elif h == 2:
            return "a very cold and dry day"
        elif h == 3:
            return "a very cold, humid day"
        else:
            return "a very humid, cold day"
    elif t == 2:
        if h == 1:
            return "a very cool, dry day"
        elif h == 2:
            if np.random.rand() < 0.5:
                return "a cool, dry day"
            else:
                return "a dry, cool day"
        elif h == 3:
            if np.random.rand() < 0.5:
                return "a cool, humid day"
            else:
                return "a humid, cool day"
        else:
            return "a very humid, cool day"
    elif t == 3:
        if h == 1:
            return "a very dry, warm day"
        elif h == 2:
            if np.random.rand() < 0.5:
                return "a warm, dry day"
            else:
                return "a dry, warm day"
        elif h == 3:
            return "a warm, humid day"
        else:
            return "a very humid, warm day"
    else:
        if h == 1:
            return "a very dry and hot day"
        elif h == 2:
            return "a very hot and dry day"
        elif h == 3:
            return "a very hot, humid day"
        else:
            return "a very humid, hot day"


# Returns two sentences - S1 <EOS> S2 with two elements in one of them and one in the other.
# Input data : [temp, humid, wind, prec, day], [bool, bool, bool, bool, True]
# Output data : S1 <EOS> S2
def split_two_twogenerator(data, choosers):
    # TODO(bitesandbytes) : Split into two sentences.
    return all_in_one_generator(data, choosers)


# Returns one sentence - S1 with all elements in it.
# Input data : [temp, humid, wind, prec, day], [bool, bool, bool, bool, True]
# Output data : S1
def all_in_one_generator(data, choosers):
    day = ""
    if data[4] == 0:
        day = "today"
    else:
        day = "tomorrow"

    temp_humid = _temp_and_humid(data[0], data[1])
    wind_prec = _wind_and_prec(data[2], data[3])

    if np.random.rand() < 0.5:
        return day + " is going to be " + temp_humid + " " + wind_prec + "."
    else:
        return "it is going to be " + temp_humid + " today " + wind_prec + "."


# Generates a random data point <T,H,W,P,D>
def _random_points():
    vec = np.random.rand(5).tolist()
    if vec[4] > 0.5:
        vec[4] = 0.
    else:
        vec[4] = 1.

    return vec


# Generates a random chooser <bool, bool, bool, bool, True>
def _random_choosers():
    vec = np.random.rand(5) > 0.5
    vec[4] = True

    return vec


generators = [split_two_twogenerator, all_in_one_generator]

if __name__ == "__main__":
    num_sentences = 50000

    strs = []
    xs = []
    choosers = []
    for _ in range(num_sentences):
        x = _random_points()
        chooser = _random_choosers()
        xs.append(x)
        choosers.append(chooser)
        strs.append(random.choice(generators)(x, chooser))

    f1 = file("../data/complex_targets_" + str(num_sentences) + ".txt", "w")
    f2 = file("../data/complex_xs_" + str(num_sentences) + ".txt", "w")

    for n in range(len(strs)):
        f1.write(strs[n] + "\n")
        f2.write("%0.4f %0.4f %0.4f %0.4f %0.4f\n" % (xs[n][0], xs[n][1], xs[n][2], xs[n][3], xs[n][4]))

    f1.close()
    f2.close()
