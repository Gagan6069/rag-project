class ScoreFilter:

    @staticmethod
    def filter(results, threshold):

        filtered = []

        for doc, score in results:

            if score <= threshold:
                filtered.append((doc, score))

        return filtered