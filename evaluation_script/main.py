import random

from datasets import load_metric


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']
    """
    dataset = json.load(open(test_annotation_file, "r"))
    qa_references, qr_references = [], []
    for qa in dataset:
        qa_references.append(
            {
                "id": "%d_%d" % (qa["Conversation_no"], qa["Turn_no"]),
                "answer": qa["Answer"]
            }
        )
        qr_references.append(
            {
                "id": "%d_%d" % (qa["Conversation_no"], qa["Turn_no"]),
                "rewrite": qa["Rewrite"]
            }
        )

    # TODO how to differentiate between submissions?


    # calculate QA metrics
    qa_predictions = json.load(open(user_submission_file, "r"))
    assert (len(qa_predictions) == len(qa_references))
    qa_metric = load_metric("squad_v2")
    qa_metric.add_batch(predictions=qa_predictions, references=qa_references)
    qa_score = qa_metric.compute()
    # OrderedDict([('exact', 33.333333333333336), ('f1', 38.095238095238095), ('span', 33.333333333333336), ('total', 3), ('HasAns_exact', 33.333333333333336), ('HasAns_f1', 38.095238095238095), ('HasAns_total', 3)])


    # calculate QR metrics
    qr_predictions = json.load(open(user_submission_file, "r"))
    assert (len(qr_predictions) == len(qr_references))
    qr_metric = load_metric("rouge")
    qr_metric.add_batch(predictions=qr_predictions, references=qr_references)
    qr_score = qr_metric.compute()
    # rouge1 rougeL

    output = {}
    if phase_codename == "original":
        print("Evaluating for Original Test Set Phase")
        output["result"] = [
            {
                "test_split": {
                    "EM": qa_score['exact'],
                    "F1": qa_score['f1'],
                    "ROUGE-1 R": qr_score['rouge1'].mid.recall
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
