import random
import json
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
    qa_ids, qr_ids = [], []
    for qa in dataset:
        _id = "%d_%d" % (qa["Conversation_no"], qa["Turn_no"])
        # skip empty answers and rewrites
        if qa["Answer"]:
            qa_ids.append(_id)
            qa_references.append(
                {
                    "id": _id,
                    "answers": {'answer_start': [0], 'text': [qa["Answer"]]}
                }
            )
        if qa["Rewrite"]:
            qr_ids.append(_id)
            qr_references.append(qa["Rewrite"])

    # TODO how to differentiate between submissions?
    print(user_submission_file)
    submission = json.load(open(user_submission_file, "r"))
    
    qa_predictions, qr_predictions = [], []
    for qa in submission:
        _id = "%d_%d" % (qa["Conversation_no"], qa["Turn_no"])
        if "Model-Answer" in qa and _id in qa_ids:
            if not qa["Model-Answer"]:
                qa["Model-Answer"] = ""
            qa_predictions.append(
                {
                    "id": _id,
                    "prediction_text": qa["Model-Answer"],
                    'no_answer_probability': 0.
                }
            )
        if "Model-Rewrite" in qa and _id in qr_ids:
            if not qa["Model-Rewrite"]:
                qa["Model-Rewrite"] = ""
            qr_predictions.append(qa["Model-Rewrite"])

    if qa_predictions:
        # calculate QA metrics
        assert (len(qa_predictions) == len(qa_references))
        qa_metric = load_metric("squad_v2")
        print(qa_predictions[0])
        print(qa_references[0])

        qa_metric.add_batch(predictions=qa_predictions, references=qa_references)

        qa_score = qa_metric.compute()
        # OrderedDict([('exact', 33.333333333333336), ('f1', 38.095238095238095), ('span', 33.333333333333336), ('total', 3), ('HasAns_exact', 33.333333333333336), ('HasAns_f1', 38.095238095238095), ('HasAns_total', 3)])
        qa_results = {
                        "EM": qa_score['exact'],
                        "F1": qa_score['f1'],
                     }

    if qr_predictions:
        # calculate QR metrics
        assert (len(qr_predictions) == len(qr_references))
        qr_metric = load_metric("rouge")
        qr_metric.add_batch(predictions=qr_predictions, references=qr_references)
        qr_score = qr_metric.compute()
        # rouge1 rougeL
        qr_results = {
                        "ROUGE-1 R": qr_score['rouge1'].mid.recall
                     }

    output = {}
    if phase_codename == "original":
        print("Evaluating for Original Phase")
        output["result"] = [{}]
        if qa_predictions:
            output["result"][0]["original_test_set_question_answering"] = qa_results
        if qr_predictions:
            output["result"][0]["original_test_set_question_rewriting"] = qr_results
               
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print(output["submission_result"])
        print("Completed evaluation for Original Phase")
    return output
