 
from skimage import measure, morphology
import numpy as np
from  scipy import ndimage 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report, ConfusionMatrixDisplay
from scipy.stats import wasserstein_distance
from ipywidgets import interact


def filter_regions_by_volume(labeled_image, volume_threshold1,volume_threshold2):   
   labels, volumes = np.unique(labeled_image, return_counts=True)
   tf_v1=volumes[volumes < volume_threshold1]
   tf_l1=labels[volumes < volume_threshold1]

   tf_v2=volumes[volumes > volume_threshold2]
   tf_l2=labels[volumes > volume_threshold2]  

   final_tf_v=np.concatenate((tf_v1,tf_v2),axis=0)
   final_tf_l=np.concatenate((tf_l1,tf_l2),axis=0)
   # print(final_tf_v)
   # print(final_tf_l)

   labels_to_remove = final_tf_l
   # Create a mask of regions to remove
   mask_to_remove = np.isin(labeled_image, labels_to_remove)

   # Set the regions below the threshold to background
   labeled_image_filtered = labeled_image.copy()
   labeled_image_filtered[mask_to_remove] = 0
   labeled_image_int = ndimage.label(labeled_image_filtered > 0)[0]

   labels, volumes = np.unique(labeled_image_int, return_counts=True)
   labels_to_remove = labels[volumes > volume_threshold2]
    # Create a mask of regions to remove
   mask_to_remove = np.isin(labeled_image_int, labels_to_remove)

    # Set the regions below the threshold to background
   labeled_image_filtered = labeled_image_int.copy()
   labeled_image_filtered[mask_to_remove] = 0
   labeled_image_final = ndimage.label(labeled_image_filtered > 0)[0]
   return(labeled_image_final)

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def random_forest_classifier(data,columns_drop,column_test):
    
    X=data.drop(columns_drop,axis=1)
    y=data[column_test]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)


    rf_clf = RandomForestClassifier(n_estimators=500,max_features=100, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_clf, X_res, y_res, cv=kfold, scoring='accuracy')
    cv_predictions = cross_val_predict(rf_clf, X_res, y_res, cv=kfold)

    rf_clf.fit(X_res, y_res)
    y_pred = rf_clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred,normalize="true")

    average_cv_accuracy = np.mean(cv_scores)

    feature_importances = rf_clf.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
 
    return(conf_matrix,report,importance_df,cv_scores,test_accuracy)

def random_forest_classifier_leave_one_out(train,test,columns_drop,column_test):
    
    X_train=train.drop(columns_drop,axis=1)
    y_train=train[column_test]

    X_test=test.drop(columns_drop,axis=1)
    y_test=test[column_test]


#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(np.shape(X_res))

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_clf, X_res, y_res, cv=kfold, scoring='accuracy')
    cv_predictions = cross_val_predict(rf_clf, X_res, y_res, cv=kfold)

    rf_clf.fit(X_res, y_res)
    y_pred = rf_clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred,normalize="true")

    average_cv_accuracy = np.mean(cv_scores)

    feature_importances = rf_clf.feature_importances_
    feature_names = X_train.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
 
    return(conf_matrix,report,importance_df,cv_scores,test_accuracy)

def wasserstein_distance_df(df1,df2):
    wasserstein_distances = {}
    for column in df1.columns:
        wasserstein_distances[column] = wasserstein_distance(df1[column], df2[column])
        
    # Convert the results to a DataFrame for better visualization
    wasserstein_df = pd.DataFrame(list(wasserstein_distances.items()), columns=['Column', 'Wasserstein Distance'])
    return(wasserstein_df)


from sklearn.model_selection import train_test_split

def uniform_and_split_images(size_image,nchannels,raw_images_cleaned,raw_image_id,where_index,seed,test,val,minmaxscale=True):
    res=np.zeros((len(raw_images_cleaned),nchannels,size_image,size_image))
    
    for r in range(len(raw_images_cleaned)):
        imagerc=raw_images_cleaned[r]
        imagerc=np.max(imagerc,0)
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
            
            res[r,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
    rid=raw_image_id
    wid=where_index
    print(len(wid))
    imTrain,imValTest,ridTrain,ridValTest,widTrain,widValTest=train_test_split(res,rid,wid,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,ridVal,ridTest,widVal,widTest=train_test_split(imValTest,ridValTest,widValTest,test_size=test/(val+test), random_state=seed, shuffle=True)

    return imTrain,imVal,imTest,ridTrain,ridVal,ridTest,widTrain,widVal,widTest
