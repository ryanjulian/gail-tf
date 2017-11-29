grep EpRewMean training.log | awk '{print $4}' | feedgnuplot --points --legend 0 "data 0" --title "EpRewMean" --y2 1 --terminal 'dumb 60,30' --exit
grep EpLenMean training.log | awk '{print $4}' | feedgnuplot --points --legend 0 "data 0" --title "EpLenMean" --y2 1 --terminal 'dumb 60,30' --exit
