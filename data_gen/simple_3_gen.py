# A simple data generator file with 3 params - (temp, wind, humidity)

import bisect
import random

# Temperature phrases & upper bounds
temp_phrases = ["It is going to be " + s + "." for s in
                ["a cold day today", "a cool day today", "a warm day today", "a hot day today"]];
temp_upper_bounds = [0.25, 0.50, 0.75, 1.0];

# Wind phrases & upper bounds
wind_phrases = ["It is going to be " + s + "." for s in
                ["a very still day today", "a calm day today", "a windy day today", "a cyclonic day today"]]
wind_upper_bounds = [0.25, 0.50, 0.75, 1.0]

# Humidity phrases & upper bounds
humid_phrases = ["It is going to be " + s + "." for s in
                 ["a very dry day today", "a dry day today", "a humid day today", "a very humid day today"]]
humid_upper_bounds = [0.25, 0.50, 0.75, 1.0]

if __name__ == "__main__":
    op_file = "data/targets"
    ip_file = "data/xs"
    num_paragraphs = 1000
    f = open(op_file + str(num_paragraphs) + ".txt", "w")
    f2 = open(ip_file + str(num_paragraphs) + ".txt", "w")

    for k in range(num_paragraphs):
        # Generate three random numbers
        rand_temp = random.uniform(0, 1)
        rand_wind = random.uniform(0, 1)
        rand_humid = random.uniform(0, 1)

        # Obtain selector idxs
        temp_idx = bisect.bisect(temp_upper_bounds, rand_temp)
        wind_idx = bisect.bisect(wind_upper_bounds, rand_wind)
        humid_idx = bisect.bisect(humid_upper_bounds, rand_humid)

        # Obtain phrases
        temp = temp_phrases[temp_idx]
        wind = wind_phrases[wind_idx]
        humid = humid_phrases[humid_idx]

        # Print concatenated paragraph to STDOUT
        paragraph = temp + " " + wind + " " + humid
        f.write(paragraph + "\n")
        f2.write("%0.4f %0.4f %0.4f\n" % (rand_temp, rand_wind, rand_humid))

    f.close()
    f2.close()
