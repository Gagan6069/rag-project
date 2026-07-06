from evaluation.evaluator import RAGEvaluator


def load_questions(file_path):

    with open(file_path, "r", encoding="utf-8") as f:

        return [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]


if __name__ == "__main__":

    evaluator = RAGEvaluator()

    questions = load_questions("evaluation/questions.txt")

    evaluator.run(questions)