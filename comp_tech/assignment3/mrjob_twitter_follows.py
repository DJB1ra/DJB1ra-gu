#!/usr/bin/env python3

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJobTwitterFollows(MRJob):

    def mapper(self, _, line):
        # Split the line at the colon
        parts = line.strip().split(':')
        id_ = parts[0]
        follows_acc = parts[1].split()  # Split by space, assuming IDs are separated by space
        # If there are two parts (ID and whatever follows), yield the ID and count of follows
        if len(parts) == 2:
            yield (id_, len(follows_acc))

    def reducer(self, id_, counts):
        # Sum the counts for each ID
        yield (None, (id_, sum(counts)))

    def reducer2(self, _, followcounts):

        max_follows_count = (None, -1)
        total_follows = 0
        unique_ids_count = 0  # determine count of unique ids
        no_follows_count = 0  # Counter for IDs with no followers
        for id_, count in followcounts:
            total_follows += count
            if count == 0:
                no_follows_count += 1
            if count > max_follows_count[1]:            
                max_follows_count = (id_, count)
            unique_ids_count += 1
        yield ('max', max_follows_count)
        yield ('Follows no account', no_follows_count)  # Yield count of IDs with no followers
        yield ('avg', total_follows / unique_ids_count)


    def steps(self):
        return [MRStep(mapper=self.mapper,
                        reducer=self.reducer),
                    MRStep(reducer=self.reducer2)]

if __name__ == '__main__':
    MRJobTwitterFollows.run()
