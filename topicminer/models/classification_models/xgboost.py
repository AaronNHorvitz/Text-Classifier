from xgboost import XGBClassifier

class AutoLabelEncoder:
    """
    Provides an automatic mechanism for encoding and decoding label data, 
    simplifying the handling of categorical labels for machine learning models. 
    This class is a wrapper around the `LabelEncoder` from scikit-learn, adding 
    functionality to automatically check if labels are numeric and only transform 
    them if they are not.

    Attributes
    ----------
    encoder : LabelEncoder
        A `LabelEncoder` object from scikit-learn used for encoding label data.
    is_encoded : bool
        Flag indicating whether the encoder has been fitted and labels have been 
        transformed.

    Methods
    -------
    fit_transform(y)
        Fits the label encoder and transforms labels to numeric form if they are 
        not already numeric.
    transform(y)
        Transforms labels using the fitted encoder if they have been previously 
        encoded.
    inverse_transform(y)
        Converts numeric labels back to their original form if they have been 
        previously encoded.
    is_numeric(y)
        Static method to check if the label data is numeric.

    Examples
    --------
    >>> labels = ['apple', 'banana', 'apple', 'orange']
    >>> encoder = AutoLabelEncoder()
    >>> numeric_labels = encoder.fit_transform(labels)
    >>> print(numeric_labels)
    [0 1 0 2]
    >>> original_labels = encoder.inverse_transform(numeric_labels)
    >>> print(original_labels)
    ['apple', 'banana', 'apple', 'orange']
    """

    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_encoded = False

    def fit_transform(self, y):
        """
        Fit label encoder and transform labels to numeric format if not already numeric.

        Parameters
        ----------
        y : array-like
            The label data to encode. Can be any sequence-like object that can 
            be converted to an array.

        Returns
        -------
        ndarray
            Array of transformed labels in numeric format.

        Notes
        -----
        If the labels are already numeric, this method does not modify them.
        The encoder remembers the labels and can be used later for inverse transformations.
        """
        if not self.is_numeric(y):
            y = self.encoder.fit_transform(y)
            self.is_encoded = True
        return y

    def transform(self, y):
        """
        Transform labels using the already fitted encoder.

        Parameters
        ----------
        y : array-like
            Labels to transform. Must be the same as or a subset of the labels
            used during fit_transform.

        Returns
        -------
        ndarray
            Array of transformed labels in numeric format if previously encoded;
            otherwise, returns the input array as is.

        Raises
        ------
        ValueError
            If trying to transform without first calling `fit_transform`.
        """
        if self.is_encoded:
            return self.encoder.transform(y)
        return y

    def inverse_transform(self, y):
        """
        Convert numeric labels back to original labels if they were previously transformed.

        Parameters
        ----------
        y : array-like
            Numeric labels to convert back to original labels.

        Returns
        -------
        ndarray
            Array of original labels.

        Raises
        ------
        ValueError
            If trying to inverse transform without first calling `fit_transform`.
        """
        if self.is_encoded:
            return self.encoder.inverse_transform(y)
        return y

    @staticmethod
    def is_numeric(y):
        """
        Determine if the label data is already in numeric format.

        Parameters
        ----------
        y : array-like
            Label data to check.

        Returns
        -------
        bool
            True if the data type of the labels is an integer type; False otherwise.
        """
        return issubclass(y.dtype.type, np.integer)


def train_xgb_classifier(X_train_bal, y_train_bal, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    Trains an XGBoost Classifier with the specified parameters.

    Parameters
    ----------
    X_train_bal : array-like of shape (n_samples, n_features)
        The training input samples, ideally balanced.
    y_train_bal : array-like of shape (n_samples,)
        The target labels for the training input samples.
    n_estimators : int, optional
        The number of gradient boosted trees. Equivalent to the number of boosting rounds.
    max_depth : int, optional
        Maximum tree depth for base learners.
    learning_rate : float, optional
        Boosting learning rate (xgb’s “eta”)

    Returns
    -------
    XGBClassifier
        The trained XGBoost classifier.
    """
    # Initialize the XGBoost classifier with provided parameters
    xgb_clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,  # To avoid a deprecation warning from XGBoost
        eval_metric='mlogloss'  # Multi-class log loss for evaluation
    )

    # Fit the model on the balanced training data
    xgb_clf.fit(X_train_bal, y_train_bal)

    return xgb_clf

