 #!/usr/bin/env python3

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJobTwitterFollows(MRJob):

    def mapper(self, key, value):
        value = value.split(":")
        yield(value[0], len(value[1].split()))


    def reducer(self, c, counts):
        yield(None ,(c,sum(counts)))

    def combiner(self, c, ones):
        yield(c, sum(ones))

    def reducer2(self, _, fCounts):
        minimumf = (None, float('inf'))
        maximumf = (None, -1)
        numf = 0
        totalf = 0
        nof = 0

        for (c, counts) in fCounts:
            numf += 1
            totalf += counts
            if counts < minimumf[1]:
                minimumf = (c, counts)
            if counts > maximumf[1]:
                maximumf = (c, counts)
            if counts == 0:
                nof += 1
        
        yield ('most followed id', maximumf[0])
        yield ('most followed', maximumf[1])
        yield ('average followed', totalf/numf)
        yield ('count follows no-one', nof)


    def steps(self):
        return [MRStep(mapper=self.mapper, combiner = self.combiner, reducer=self.reducer), MRStep(reducer=self.reducer2)]

if __name__ == '__main__':
    MRJobTwitterFollows.run()
