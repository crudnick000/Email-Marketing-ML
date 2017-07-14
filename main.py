
import sys


_tool = sys.argv[1].lower()


# Spam Classifier
if _tool == "spam":

    # Command Line Format ( "spam"  (nnet / baye)  (train / test)  model_dir )
    _model = sys.argv[2].lower()
    _method = sys.argv[3].lower()
    _dir = sys.argv[4].lower()
    _modelFile = sys.argv[5].lower()

    if _model == "nnet":
        import spam_nn
        #spam_nn.main(_method, _dir, _modelFile)

    elif _model == "baye":
        import spam_bn
        spam_bn.main(_method, _dir, _modelFile)

    else:
        sys.exit("Main Error: invalid method.")


# Linear Regression (Subject Line -> Open Rate)
elif _tool == "subject_line":

    # Command Line Format ( "subject_line"  (train / test)  model_dir )
    _method = sys.argv[2].lower()
    _dir = sys.argv[3].lower()

    if _method == "least_squared":
        pass

    elif _method == "grad_descent":
        pass

    else:
        sys.exit("Main Error: invalid method.")


# Linear Regression (Subject Line -> Open)
elif _tool == "email":

    # Command Line Format ( "email"  (train / test)  model_dir )
    _method = sys.argv[2].lower()
    _dir = sys.argv[3].lower()

    if _method == "least_squared":
        pass

    elif _method == "grad_descent":
        pass

    else:
        sys.exit("Main Error: invalid method.")


# Logistic Regression (Email Content -> Click Through Rate)
elif _tool == "personalize":

    # Command Line Format ( "personalize"  id )
    _id = sys.argv[2].lower()


else:
    sys.exit("Main Error: invalid tool.")



