#Function to calculate the corresponding statistics for the classifier:
def cal_score(Y_label, scoring):
    #AUROC/ AUPR calculation:
    fpr, tpr, _ = roc_curve(Y_label, scoring)
    auroc = roc_auc_score(Y_label, scoring)
    pr, rec, _ = precision_recall_curve(Y_label, scoring)
    aupr = average_precision_score(Y_label, scoring)
    #####
    plt.step(fpr, tpr, color = 'blue')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('AUC is {:.6f}'.format(auroc))
    plt.show()
    
    plt.step(rec, pr, color = 'b')
    plt.xlabel('PR')
    plt.ylabel('REC')
    plt.title('AUPR is {:.6f}'.format(aupr))
    
    #Transform the scoring to binary prediction:
    label_pred = [1 if t > 0.5 else 0 for t in scoring]
    
    #Output confusion matrix:
    conf_mat = confusion_matrix(Y_label, label_pred)
    
    #Evaluation the for label classification:
    accu = accuracy_score(Y_label, label_pred)
    prec = precision_score(Y_label, label_pred)
    recc = recall_score(Y_label, label_pred)
    kap = cohen_kappa_score(Y_label, label_pred)
    #Evaluate the MCC:
    mcc = matthews_corrcoef(Y_label, label_pred)
    
    #Generate report:
    report = pd.DataFrame({'auc': [auroc], 'aupr': [aupr], 'accu': [accu], 'prec': [prec], 'recc': [recc], 'kap': [kap], 'mcc':[mcc]})
    
    return report, conf_mat