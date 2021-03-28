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
    # acknowledgement: https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/sharedtask_utils.py
    metric = load_metric("squad_v2")

    dataset = json.load(open(test_annotation_file, "r"))
    references = []
    for qa in dataset:
        references.append(
            {
                "id": "%d_%d" % (qa["Conversation_no"], qa["Turn_no"]),
                "answers": qa["Answer"],
            }
        )

    predictions = json.load(open(user_submission_file, "r"))

    assert (len(predictions) == len(references))

    metric.add_batch(predictions=predictions, references=references)
    final_score = metric.compute()
    # OrderedDict([('exact', 33.333333333333336), ('f1', 38.095238095238095), ('span', 33.333333333333336), ('total', 3), ('HasAns_exact', 33.333333333333336), ('HasAns_f1', 38.095238095238095), ('HasAns_total', 3)])

    output = {}
    if phase_codename == "original":
        print("Evaluating for Original Test Set Phase")
        output["result"] = [
            {
                "test_split": {
                    "EM": final_score['exact']
                    "F1": final_score['f1']
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
