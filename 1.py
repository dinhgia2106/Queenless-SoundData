start_time = time.time()

lr_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, 
                                   class_weight=None, random_state=42, solver='lbfgs', max_iter=1500, multi_class='deprecated', 
                                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
lr_classifier.fit(train_features, train_labels)

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

val_pred_lr = lr_classifier.predict(val_features)
test_accuracy_lr = lr_classifier.predict(test_features)
print(f"Logistic Regression (FFT features) - Validation Accuracy: {accuracy_score(val_labels, val_pred_lr)*100:.2f}%")
print(f"Logistic Regression (FFT features) - Test Accuracy: {accuracy_score(test_labels, test_accuracy_lr)*100:.2f}%")