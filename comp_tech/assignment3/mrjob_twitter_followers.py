from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJobTwitterFollowers(MRJob):
    
    def mapper(self, _, line):
        # Split the line at the colon
        parts = line.strip().split(':')
        id_ = parts[0]
        follows = parts[1].split()  # Split by space to get the list of followed IDs
        
        # Emit each followed ID with a count of 1 for each occurrence
        yield (id_, 0)

        for followed_id in follows:
            yield followed_id.strip(), 1

    def combiner(self, key, values):
        # Sum the occurrences of each ID
        yield key, sum(values)

    def reducer(self, id, counts):
        followers = sum(counts)
        yield None, (id, followers)

    def reducer2(self, _, followers):
        
        no_followers_count = 0
        max_followers = (None, -1)
        unique_id_count = 0
        total_followers = 0

        for user, numfollowers in followers:
            unique_id_count += 1
            total_followers += numfollowers
            if numfollowers == 0:
                no_followers_count += 1
            if numfollowers>max_followers[1]:
                max_followers = (user, numfollowers)

        average_followers = total_followers / unique_id_count

        yield('User with most followers:', max_followers)
        yield('Average followers', average_followers)
        yield('Count of no followers', no_followers_count)

    def steps(self):
        return [MRStep(mapper=self.mapper,combiner=self.combiner,
                        reducer=self.reducer), 
                MRStep(reducer=self.reducer2)]

if __name__ == '__main__':
    MRJobTwitterFollowers.run()
