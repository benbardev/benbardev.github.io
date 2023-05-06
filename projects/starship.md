## Starship Titanic

This project is based on solving the Kaggle competition problem Starship Titanic. The main Kaggle competition page can be found [here](https://www.kaggle.com/competitions/spaceship-titanic/overview)

Summary: To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

My code uses an ensamble of a Random Forest and a Neural Network. The data was cleaned by removing columns that did not improve the model. Gaps in the dataset were iteratively filled using predictions from the Random Forest.

<img src="images/starship_importance.png" style="border-radius:5%">