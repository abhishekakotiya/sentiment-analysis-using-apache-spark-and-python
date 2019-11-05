from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    timestep = []
    time = 0
    positive = []
    negative = []
    
    for count in counts:
    	timestep.append(time)
    	time += 1
    	positive.append(count[0][1])
    	negative.append(count[1][1])
    
    plt.xlabel('Time step')
    plt.ylabel('Word count')
    plt.xticks(timestep)
    plt.plot(timestep, positive, marker='o', color='b', label='positive')
    plt.plot(timestep, negative, marker='o', color='g', label='negative')
    plt.legend()
    plt.show()
    

def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    wordFile = open(filename)
    wordList = [word.strip() for word in set(wordFile.readlines())]
    wordFile.close()
    return set(wordList)


def keepRunningTotal(wtypeCounter, runningTotal):
	if runningTotal is None:
		runningTotal = 0
	return sum(wtypeCounter, runningTotal)


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    words = tweets.flatMap(lambda tweet: tweet.split(" "))
    # filter words which are not in pwords or nwords
    pnwords = pwords.union(nwords)
    words = words.filter(lambda word: word in pnwords)    

    pos_neg = words.map(lambda word: ('positive', 1) if word in pwords else ('negative', 1))
    pos_negCount = pos_neg.reduceByKey(lambda count1, count2: count1 + count2)
    runningTotal = pos_neg.updateStateByKey(keepRunningTotal)
    
    runningTotal.pprint()
       
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    pos_negCount.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
