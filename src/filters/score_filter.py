class ScoreFilter:

    @staticmethod
    def filter(results, threshold):

        if threshold is None:
            return results

        filtered = []

        for result in results:

            if result.score <= threshold:
                filtered.append(result)

        return filtered