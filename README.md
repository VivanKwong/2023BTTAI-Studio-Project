# 2023BTTAI-Studio-Write-Up
Smart Restaurant Recommendation System, Dropbox
Fall 2023, AI Studio Project Write-Up

Project Description: The goal of this project is to develop a restaurant recommendation system that will suggest restaurants for users to attend based on the information they input. Since our model suggests restaurants, this is a supervised learning and classification problem. This is also an unsupervised learning problem using clustering of similar users and restaurants.
Table of Contents
	Business Focus
	Data Preparation and Validation
	Approach
	Key Findings and Insights
	Acknowledgements

Business Focus
The principles used and developed for this project, such as methodology and model and evaluating performance skills, are generally applicable to other recommendation system projects. Even though this is a representative problem and Dropbox doesn’t need a restaurant recommendation system in particular, the principles that we’ll learn from this project are very transferable. Dropbox also currently has a feature that recommends files to users.
Data Preparation and Validation
Dataset Description: We were given nine data files by Dropbox, sourced from Kaggle as “Restaurant Data with Consumer Ratings”:
Restaurant Accepted Mode(s) of Payment
Restaurant Type(s) of Cuisine
Restaurant Hours
Restaurant Parking
Restaurant Profiles
User Preferred Mode(s) of Payment
User Preferred Type(s) of Cuisine
User Profiles
User Ratings
Not all data files were needed for the project. We decided to focus on using “Restaurant Type(s) of Cuisine”, “Restaurant Profiles”, “User Preferred Type(s) of Cuisine”, “User Profiles”, and “UserRatings”. We also decided to make the ratings in “UserRatings” more readable by making the ratings range the current rating range of 0 to 2 to a range of 1 to 3, and named this new file “UserRestaurantRatings”.
EDA (Exploratory Data Analysis): While exploring the data files, we noticed at first glance that the data isn’t very clean, and that a lot of time was going to be spent dealing with outliers, incorrect data, missing data (we also had to decide, were we going to delete them or impute them?). During a meeting with our Challenge Advisors, we asked for advice about the process. Ameya suggested that we put some of the data in an Excel spreadsheet and look at it there, explore the data programmatically with Python (NumPy and Pandas), look at what others have done already in notebooks on Kaggle, and utilize ChatGPT to analyze the datasets and inspire us. April suggested that we explore ChatGPT for suggestions, sample subsets of the data before using all of it for modeling, and use Python to combine the datasets and look out for duplicates across data files. 

Using their suggestions we were able to paint a picture with the data given. I used Folium to map the coordinates of the restaurants and users on an interactive map (images shown above). From that, we found that the data came from three cities in Mexico, and the restaurants were along a couple streets in each city. We also took note of some of the variables/features of the datasets: cuisine, payment method employed by the restaurant, whether the restaurant offers parking, location of the restaurant, hours and days of operation, price range, food and service rating, and accessibility. While we already knew that there was missing data we found a specific challenge with the discrepancies in the different restaurants across datasets: There are different pieces of information available for different restaurants. In particular, we found that: 
chefmozaccepts has information about 615 restaurants
chefmozcuisine has information about 769 restaurants
chefmozhours4 has information about 694 restaurants
chefmozparking has information about 675 restaurants
geoplaces2 has information about 130 restaurants
	

We realized that the data we could actually use for testing and training our model was smaller than we had hoped. There were a total of 1,161 user ratings for the overall quality, the food quality, and the service quality of 130 restaurants; we focused primarily on the overall ratings
Some users rated as few as 3 restaurants, while others rated as many as 18 restaurants, while some restaurants had as few as 3 ratings, while others had as many as 36 ratings. We had to look for additional data to offset missing data, identify restaurants that appear across multiple datasets, and merge on those restaurants.
It is also at this point that we decided to split into two subgroups. One group would work on content-based filtering, and the second group would work on collaborative filtering. I worked on content-based filtering with Maame Andoh and Sandy Zhang.
Feature Selection: For the content-based filtering group, feature selection took up a significant portion of our time. We noticed that there were some differing features between the different user datasets, as well as differing features between the different restaurant datasets. It was necessary for us to select relevant features from user and restaurant data that could be found across the datasets, and with few missing values. On the restaurant cuisine data side, we discovered that there were over 50 different cuisines, which was far too many. We discussed this with our Challenge Advisor and TA, who suggested combining restaurant cuisines into more general groups to make it more manageable.
# change values in price, alcohol, ambience, dress code, and smoking column for restaurants
restaurant_price = pd.read_csv('RestaurantsGeographics.csv')
restaurant_price['price'] = restaurant_price['price'].map({'low': 1, 'medium': 2, 'high': 3})
restaurant_price['alcohol'] = restaurant_price['alcohol'].map({'No_Alcohol_Served': 0, 'Wine-Beer': 1, 'Full_Bar': 1})
restaurant_price['Rambience'] = restaurant_price['Rambience'].map({'quiet': 0, 'familiar': 1})
restaurant_price['dress_code'] = restaurant_price['dress_code'].map({'informal': 0, 'casual': 0, 'formal': 1})
restaurant_price['smoking_area'] = restaurant_price['smoking_area'].map({'none': 0, 'not permitted': 0,'section': 1,'only at bar': 1,'permitted': 1})


​​# create restaurant dataframe with selected features
merged_restaurant = grouped_restaurant.merge(restaurant_price[['placeID','price', 'latitude', 'longitude', 'alcohol', 'Rambience', 'dress_code', 'smoking_area']], on='placeID', how='inner')


# change values in budget, alcohol, ambience, dress preferences, and smoking habit columns for users
user_budget = pd.read_csv('UserProfiles.csv')
user_budget['budget'] = user_budget['budget'].map({'?': 1, 'low': 1, 'medium': 2, 'high': 3})
user_budget['drink_level'] = user_budget['drink_level'].map({'abstemious': 0, 'social drinker': 1, 'casual drinker': 1})
user_budget['ambience'] = user_budget['ambience'].map({'?': 0, 'solitary': 0, 'family': 1, 'friends': 1})
user_budget['dress_preference'] = user_budget['dress_preference'].map({'?': 0, 'no preference': 0, 'informal': 0, 'elegant': 1, 'formal': 1})
user_budget['smoker'] = user_budget['smoker'].map({'FALSE': 0, 'TRUE': 1, '?': 1})


# Create user dataframe with selected features
merged_user = grouped_user.merge(user_budget[['userID','budget', 'latitude', 'longitude', 'drink_level', 'ambience', 'dress_preference', 'smoker']], on='userID', how='inner')

From the restaurant data, we chose the features “price”, “alcohol”, “Rambiance”, “dress_code”, and “smoking_area”. From the user data, we chose the features “budget”, “drink_level”, “ambiance”, “dress_preference”, and “smoker”. These features could be found in both user and restaurant data to be used to determine the compatibility later for our model. Also, in our opinion, these features were influential in users’ restaurant leanings. I label-encoded each of these features so that their possible values corresponded with a number. For example, for “drink_level”, I mapped 'abstemious' to a value of 0, 'social drinker' to a value of 1, and 'casual drinker' to a value of 1. 
Similarly, we combined restaurant cuisines into groups. We narrowed down the number of 
cuisine_mapping = {
   'Fast_Food': ['Hot_Dogs', 'Burgers', 'Pizzeria', 'Fast_Food', 'Doughnuts', 'Cafeteria', 'Deli-Sandwiches', 'Soup', 'Game'],
   '21+': ['Bar', 'Bar_Pub_Brewery'],
   'American': ['American', 'Pacific-Northwest', 'Contemporary', 'Family', 'Diner', 'Barbecue', 'Tex-Mex', 'Steaks', 'Canadian', 'Southwestern', 'Cajun-Creole', 'California', 'Southern'],
   'Central-Asian': ['Afghan', 'Mongolian', 'Tibetan'],
   'South-Asian': ['Indian-Pakistani'],    'East-Asian': ['Japanese', 'Sushi', 'Chinese', 'Dim-Sum', 'Korean'],    'Caribbean': ['Caribbean', 'Jamaican', 'Cuban'],'Middle-Eastern': ['Middle-Eastern', 'Lebanese', 'Turkish', 'Israeli', 'Persian', 'Armenian'],
   'Dietary': ['Vegetarian', 'Kosher', 'Vegan', 'Gluten-Free', 'Organic-Healthy', 'Seafood', 'Eclectic', 'Fine_Dining'],
   'Continental-European': ['Continental-European', 'Eastern_European', 'Polish', 'British', 'Austrian', 'Russian-Ukrainian', 'Irish', 'German', 'Romanian', 'Swiss', 'Hungarian', 'Dutch-Belgian', 'Scandinavian', 'French'],
   'Southern-European': ['Italian', 'Mediterranean', 'Portuguese', 'Spanish', 'Tapas', 'Basque', 'Greek'],
   'African': ['Moroccan', 'African', 'Ethiopian', 'Tunisian', 'North_African']
}

cuisines to nine, and employed one-hot encoding to show which cuisines the restaurants served.
Approach


Selected Models: We decided on using a Gradient-Boosted Classification Decision Tree to predict user ratings. This is because GBDT employs an adaptive learning rate during the training process, which allows the algorithm to assign different weights to each weak learner, emphasizing the importance of correcting errors made by the previous models. By combining the strength of multiple weak learners (shallow decision trees), GBDT can model complex relationships in the data. 
#perform cross join on user and restaurant dataframe
rename_dict = {}
for i in merged_restaurant.columns:
   rename_dict[i] = 'rest_'+i
merged_restaurant.rename(columns = rename_dict, inplace=True)


rename_dict = {}
for i in merged_user.columns:
   rename_dict[i] = 'user_'+i
merged_user.rename(columns = rename_dict, inplace=True)


cross_df = merged_restaurant.merge(merged_user, how = 'cross')


#make a new rating column and fill it with Nones
cross_df['rating'] = None


#fill in the ratings one by one based on the trainingSet dataframe
for i, row in cross_df.iterrows():


   userid = row['user_userID']
   restid = row['rest_placeID']


   rating = trainingSet[(trainingSet['userID'] == userid) & (trainingSet['placeID'] == restid)]


   if len(rating) != 0:
       num = rating['rating'].tolist()[0]
       cross_df.loc[i, 'rating'] = num


#make a new df based on rows that have rating values
ratings_df = cross_df[~cross_df['rating'].isnull()]


#make sure everything in the df is a number
ratings_df['rating'] = ratings_df['rating'].astype(float)
ratings_df['user_userID'] = ratings_df['user_userID'].map(lambda x: int(x[1:]))

Prior to implementing the GBDT model, we performed a cross-join on the Restaurant and User data frames and made a new dataframe based on rows with rating values (code pictured above). Next, we tuned the hyperparameters: learning rate, max depth of tree, and number of estimators. Then, we trained the gradient boosting classifier on the training set, and predicted on the validation set and testing set.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#split the train and test sets
train_df, test_df = train_test_split(ratings_df, test_size=0.2)
print(train_df.shape, test_df.shape)


#train a decision tree model
train_X = train_df.iloc[:, 0:-1]
train_y = train_df.iloc[:, -1]


test_X = test_df.iloc[:, 0:-1]
test_y = test_df.iloc[:, -1]


# Split the data into training and testing sets
# You can adjust the test_size and random_state parameters as needed
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42)


# Create the GradientBoostingClassifier
# You can adjust hyperparameters such as n_estimators, learning_rate, max_depth, etc.
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# Train the model
gb_classifier.fit(X_train, y_train)


# Predictions on the validation set
val_predictions = gb_classifier.predict(X_val)


# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)


# Print Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))


print("\nClassification Report:")
print(classification_report(y_val, val_predictions))


# Fit the model on the entire training set
gb_classifier.fit(train_X, train_y)


# Evaluate on the test set
test_predictions = gb_classifier.predict(test_X)
test_accuracy = accuracy_score(test_y, test_predictions)
print("\nTest Set Accuracy:", test_accuracy)


print(gb_classifier.score(train_X, train_y))


#look at the predictions and evaluate
pred_train_y = gb_classifier.predict(train_X)
print((abs(train_y - pred_train_y)).mean())


print(list(pred_train_y))
print(list(train_y))


pred_test_y = gb_classifier.predict(test_X)
#our model's predictions' average error
print((abs(test_y - pred_test_y)).mean())
#if we just use the average rating as the prediction
print(abs(test_y - train_y.mean()).mean())


print(list(pred_test_y))
print(list(test_y))

Preventing Overfitting: GradientBoostingClassifier from scikit-learn has some inherent features that help prevent overfitting. By tuning the hyperparameters, learning rate, max depth of tree, and number of estimators, we were able to minimize overfitting. We also prevented overfitting in our validation strategy.
Validation Strategy: The validation strategy we employed for this model is using a validation set to monitor the model's performance during training. We split the training data into two subsets: a training set (X_train, y_train) and a validation set (X_val, y_val).
train_df, test_df = train_test_split(ratings_df, test_size=0.2)

The test_size=0.2 parameter specifies that 20% of the data should be allocated for the validation set. 
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

The random_state=42 parameter makes sure that the data split is reproducible. This means that the same random seed will result in the same data split, which produces consistent results.
gb_classifier.fit(X_train, y_train)

The code above trains the model on the X_train, y_train training set using the fit method.
val_predictions = gb_classifier.predict(X_val)

The code above evaluates the model’s performance on the X_val, y_val validation set using the predict method.
To evaluate the accuracy of the model, we compare val_predictions against the true labels y_val. 
We employed a validation strategy of splitting the data because it allows us to monitor the model's performance on unseen data and helps prevent overfitting. If the model performs well on the training set but not well on the validation set, it might be overfitting the training data. This tells us to continue tuning the hyperparameters. 

The model evaluation of our Gradient Boosted Decision Tree recommendation system produced a training score of 0.839, a validation score of 0.55, and a testing score of 0.611. The training score tells us that the model correctly predicted around 83.9% of the examples in the training set. The validation score tells us that the model correctly predicted around 55% of the examples in the validation set. The testing score tells us that the model correctly predicted around 61.1% of the examples in the test set. 
Overall, our training score suggests that the model is performing well on the data it was trained on. On the other hand, the validation score might indicate that the model is not generalizing well to new, unseen data. This could be a sign of overfitting, if the model has learned the training data too well but struggles with new examples.
Key Findings and Insights
Key Results: We had hoped to combine with the results from the collaborative filtering/matrix factorization subgroup in a hybrid approach to make one model by averaging the predicted ratings for each user for each restaurant. 
While we were not able to get to this step in the time allocated, both groups were able to individually reach our goal of recommending restaurants to users.
Insights: The use of GBDT allowed us to create an algorithm that combined multiple weak learners (shallow decision trees) and used the error of previous trees to improve on the performance of following models. This produced high accuracy scores. However the decreased validation and test scores, compared to the training score, could be attributed to limited data and the ramifications of working with a small dataset. 
Lessons Learned: We learned about the difficulty of working with 1) such few ratings and 2) a lack of numeric data. We also got to experience the rewards of merging datasets with different information spread across them and the limitations of recommender systems (cold-start problem for new users). Finally, we learned about the difficulty of working with a remote team with very different schedules. For a team to work efficiently it is important that all members make an effort to join meetings and contribute. 
Potential Next Steps: In future improvements, we could cluster users using their categorical data, then run matrix factorization on user clusters for another hybrid content-based & collaborative approach. In addition, we could cluster restaurants using their categorical data, then recommend similar restaurants to those users already rated highly. Also, it could be valuable to explore using a neural network to discover more complex relationships between users and restaurants.
Acknowledgements
I express my utmost gratitude for my project-mates. We did our best to make time to meet every week and work on this project, in addition to school work and regular work and applying for internships. In particular, Maame and Sandy made extra time on Sunday mornings for additional meetings where we could code and work together. I am so thankful for such dedicated teammates.

I also am grateful for the Dropbox Challenge Advisors. Ameya and April were a valuable source of wisdom and insight. Their advice grounded and guided us.

Similarly, David Fang, our TA, was a great help as well. He always made himself available for us, even during a road trip. I do not know what we would have done without him.

Special thanks to my mentor, Weiwei, for coming to our presentation for support. 

Finally, I would like to thank the BTTAI coordinators for organizing this; I learned so much from this project and am glad to have had this opportunity.
