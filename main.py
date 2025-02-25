import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from env import DynamicPricingForECommerce
from model import QLearningAgent
import re
import matplotlib.pyplot as plt

def compute_demand(df):

    grouped_df = df.groupby("name", as_index=False).agg({
        "Actual_price": "mean",   # Average price across all listings for this product
        "Week": "first",  # Keep brand info
        #"weight_in_kg": "first"    # Or whichever columns you need
    })
    df["count_rows"] = 1
    #demand_df = df.groupby(["id", "weight_in_kg"], as_index=False)["count_rows"].sum()
    demand_df = df.groupby(["name", 'Week'], as_index=False)["count_rows"].sum()


    #grouped_df = grouped_df.merge(demand_df, on=["id", "weight_in_kg"], how="left", copy=False)
    grouped_df = grouped_df.merge(demand_df, on=["name", 'Week'], how="right", copy=False)

    #df.merge(grouped_df, on=["id", "weight_in_kg"], how = 'left', copy=False)
    df.merge(grouped_df, on=["name", "Week"], how = 'right', copy=False)
    df.rename(columns={"count_rows": "base_demand"}, inplace=True)
    return df

def compute_elasticities(df):
    Week_price_df = df.groupby(['name', 'Week']).agg({'Discount_price':'mean','Impression':'sum'}).reset_index()
    x_pivot = Week_price_df.pivot(index='Week', columns='name' ,values='Discount_price')
    x_values = pd.DataFrame(x_pivot.to_records())
    x_values.fillna(method='ffill', inplace=True)
    y_pivot = Week_price_df.pivot(index='Week', columns='name', values='Impression')
    y_values = pd.DataFrame(y_pivot.to_records())
    y_values.fillna(method='ffill', inplace=True)

    final_df = pd.DataFrame()
    for col in x_values.columns[1:]:
        results_values = {
        "name": [],
        "elasticity": [],
        "price_mean": [],
        "quantity_mean": [],
        "intercept": [],
        "t_score":[],
        "slope": [],
        "coefficient_pvalue" : [],
        "rsquared": [],
        }

        temp_df1 = pd.DataFrame()
        temp_df1['x'] = x_values[col]
        temp_df1['y'] = y_values[col]
        temp_df1.dropna(inplace=True)
        x_value = temp_df1['x']
        y_value = temp_df1['y']
        X = sm.add_constant(x_value)
        model = sm.OLS(y_value, X)
        result = model.fit()

        #choose only those whose P-value is less than 5% errornous
        if result.f_pvalue < 0.05:

            rsquared = result.rsquared
            coefficient_pvalue = result.f_pvalue
            try:
                intercept,slope = result.params
            except:    
                slope = result.params
            mean_price = np.mean(x_value)
            mean_quantity = np.mean(y_value)
            try:
                tintercept, t_score = result.tvalues
            except:
                pass
        
            #Price elasticity Formula
            price_elasticity = (slope)*(mean_price/mean_quantity)
            #Append results into dictionary for dataframe
            results_values["name"].append(col)
            results_values["elasticity"].append(price_elasticity)
            results_values["price_mean"].append(mean_price)
            results_values["quantity_mean"].append(mean_quantity)
            results_values["intercept"].append(intercept)
            results_values['t_score'].append(t_score)
            results_values["slope"].append(slope)
            results_values["coefficient_pvalue"].append(coefficient_pvalue)
            results_values["rsquared"].append(rsquared)

            final_df = pd.concat([final_df,pd.DataFrame.from_dict(results_values)],axis=0,ignore_index=True)
    return final_df

def compute_elasticities_per_name(df):
    #df["price"] = (df["prices.amountMin"] + df["prices.amountMax"]) / 2
    df["quantity"] = 1  # each row is counted as 1
    
    #grouped = df.groupby(["id", "Actual_price", "weight_in_kg"], as_index=False).agg({
    #    "quantity": "sum"
    #})
    #grouped = df.groupby(["id", "Actual_price"], as_index=False).agg({
    #    "quantity": "sum"
    #})
    grouped = df.groupby(['name', 'Week']).agg({'Actual_price':'mean','Impression':'sum'}).reset_index()
    x_pivot = grouped.pivot(index='Week', columns='name' ,values='Actual_price')
    # Prepare to store elasticity results
    product_elasticities = []
    # For each product name, run a log-log regression
    #for id, subdf in grouped.groupby(["id", "weight_in_kg"]):
    #for id, subdf in grouped.groupby(["id"]):
    for name, subdf in grouped.groupby(['name', 'Week']):
        # subdf has columns: [name, price, quantity]
        # We need at least 2 distinct price points:
        if subdf["Actual_price"].nunique() < 2:
            continue  # skip products that don't vary in price
  
        # Remove zero or negative prices/quantities
        subdf = subdf[(subdf["Actual_price"] > 0) & (subdf["quantity"] > 0)]
        if len(subdf) < 20:
            continue
        
        # Log-transform
        subdf["logQ"] = np.log(subdf["quantity"])
        subdf["logP"] = np.log(subdf["Actual_price"])
        
        # OLS: logQ ~ logP
        X = sm.add_constant(subdf["logP"])
        y = subdf["logQ"]
        
        try:
            model = sm.OLS(y, X).fit()
            alpha, beta = model.params
            # beta is the price elasticity for this product
            product_elasticities.append({
                "name": name[0],
                #"weight_in_kg": id[1],
                "alpha": alpha,
                "elasticity": beta,
                "n_points": len(subdf)  
            })
        except Exception as e:
            # If the regression fails for numerical reasons, skip
            print("OLS error")
            continue
    
    # Convert list of dicts to DataFrame
    result_df = pd.DataFrame(product_elasticities)
    #print(result_df['elasticity'].describe())
    return result_df

def Source_to_Merchant(x):
   try:
    return x.split("www.")[1].split("/")[0]
   except:
    if len(x)>0:
        return x.split("//")[1].split(".com")[0]
    return x

def preprocess(df):
    df['prices.availability'] = np.where(df['prices.availability'].str.contains('Yes|TRUE|In Stock|yes|available', flags=re.IGNORECASE), "Yes",
                 np.where(df['prices.availability'].str.contains('No|sold|FALSE|Retired|Discontinued', flags=re.IGNORECASE), "No",
                 np.where(df['prices.availability'].str.contains('Special Order|More on the Way|More Coming Soon', flags=re.IGNORECASE), "Special", ""
                 )))

    df.loc[df['prices.condition'].str.contains('new',flags=re.IGNORECASE) == True,"prices.condition"] = 'New'
    df.loc[df['prices.condition'].str.contains('refurbished',flags=re.IGNORECASE) == True,"prices.condition"] = 'Refurbished'
    df.loc[df['prices.condition'].str.contains('pre-owned|used',flags=re.IGNORECASE) == True,"prices.condition"] = 'Used'

    Impression_count=[]
    for i in df['prices.dateSeen']:
        time_= str(i).split(",")
        Impression_count.append(len(time_))

    df['Impression'] = Impression_count
    #df['prices.dateSeen'].tail()
    df['prices.dateSeen'] = df['prices.dateSeen'].apply(lambda x: x.split(",")[0]) #
    df['prices.dateSeen'] = pd.to_datetime(df['prices.dateSeen'])

    df.loc[((df['prices.amountMax'] != df['prices.amountMin']) & (df['prices.isSale'] == False)),"prices.isSale"] = True
    df.loc[((df['prices.amountMax'] == df['prices.amountMin']) & (df['prices.isSale'] == True)),"prices.isSale"] = False

    df['prices.sourceURLs'] = df['prices.sourceURLs'].apply(lambda x: Source_to_Merchant(x))
    df.loc[df['prices.merchant'].isnull(),'prices.merchant'] = df['prices.sourceURLs']

    df['prices.merchant'] = np.where(df['prices.merchant'].str.contains('bhphotovideo', flags=re.IGNORECASE), "bhphotovideo.com",
                                np.where(df['prices.merchant'].str.contains('eBay|e bay', flags=re.IGNORECASE), "ebay.com",
                 np.where(df['prices.merchant'].str.contains('Amazon',flags=re.IGNORECASE), "Amazon.com", 
                 np.where(df['prices.merchant'].str.contains('Bestbuy',flags=re.IGNORECASE), "Bestbuy.com",
                 np.where(df['prices.merchant'].str.contains('Homedepot',flags=re.IGNORECASE), "homedepot.com",
                 np.where(df['prices.merchant'].str.contains('newegg',flags=re.IGNORECASE), "newegg.com",
                 np.where(df['prices.merchant'].str.contains('kmart',flags=re.IGNORECASE), "kmart.com",
                 np.where(df['prices.merchant'].str.contains('frys',flags=re.IGNORECASE), "frys.com",
                 np.where(df['prices.merchant'].str.contains('cdw',flags=re.IGNORECASE), "cdw.com",
                 np.where(df['prices.merchant'].str.contains('target',flags=re.IGNORECASE), "target.com",
                 np.where(df['prices.merchant'].str.contains('overstock',flags=re.IGNORECASE), "overstock.com",
                 np.where(df['prices.merchant'].str.contains('barcodable',flags=re.IGNORECASE), "barcodable.com",
                 np.where(df['prices.merchant'].str.contains('kohls',flags=re.IGNORECASE), "kohls.com",
                 np.where(df['prices.merchant'].str.contains('sears',flags=re.IGNORECASE), "sears.com",
                 np.where(df['prices.merchant'].str.contains('Wal-mart|Walmart',flags=re.IGNORECASE), "Walmart.com","Other")))))))))))))))

    df['dateAdded'] = pd.to_datetime(df['dateAdded'])
    df['dateUpdated'] = pd.to_datetime(df['dateUpdated'])

    df['Country']= np.where(df['prices.currency'] == 'USD','USA',
                            np.where(df['prices.currency'] == 'CAD',"Canada",
                            np.where(df['prices.currency'] == 'SGD',"Singapore",
                            np.where(df['prices.currency'] == 'EUR',"EUROPE",
                            np.where(df['prices.currency'] == 'GBP',"UK","Other"
    )))))

    weight_list = []
    for x in df['weight'].to_list():  
        if ((x.find('lb') !=-1) or (x.find('lbs') !=-1) ):
            try:
                weight_list.append(float(x.split("lbs")[0].strip())*0.453592)
            except:
                try:
                    weight_list.append(float(x.split("lb")[0].strip())*0.453592)
                except:
                    weight_list.append(np.nan)
        elif 'pounds' in x:
            weight_list.append(float(x.split("pounds")[0].strip())*0.453592)
        elif 'ounces' in x:
            weight_list.append(float(x.split("ounces")[0].strip())*0.453592)
        elif 'oz' in x:
            weight_list.append(float(x.split("oz")[0].strip())*0.453592)
        elif (('Kg' in x) or ('kg' in x)) :
            try:
                weight_list.append(float(x.split("Kg")[0].strip()))
            except:
                weight_list.append(float(x.split("kg")[0].strip()))
        elif 'g' in x:
            try:
                weight_list.append(float(x.split("g")[0].strip())/1000)
            except:
                weight_list.append(np.nan)  
        else:
            weight_list.append(np.nan)
    df['weight_in_kg'] = weight_list


    df.rename(columns={'prices.amountMax':'Actual_price',
                        'prices.amountMin':'Discount_price',
                        'prices.availability':'Product_availability',
                        'prices.condition':'Condition',
                        'prices.currency':'Currency',
                        'prices.isSale':'isSale',
                        'prices.merchant':'merchant',
                        'prices.dateSeen':'Date',
                        'categories':'Description'
                    },inplace=True)

    Sub_df = df[['id','Actual_price','Discount_price','Product_availability','Condition','Currency','Country',
                'Date','isSale','merchant','brand','Description','primaryCategories','weight_in_kg','name','Impression']]
    Sub_df['Day'] = Sub_df['Date'].dt.day
    Sub_df['Month'] = Sub_df['Date'].dt.month
    Sub_df['Year'] = Sub_df['Date'].dt.year
    Sub_df['Week'] = Sub_df['Date'].dt.isocalendar().week
    Sub_df['Month_Name'] = Sub_df['Date'].dt.strftime('%B')
    Sub_df['Day_Name'] = Sub_df['Date'].dt.strftime('%A')
    Sub_df['Formatted_date'] = pd.to_datetime(Sub_df['Date'].dt.date)
    Sub_df.drop(columns=['Date'],inplace=True)

    return Sub_df

if __name__ == "__main__":
    file_path = "dynamic_pricing.csv"
    df = pd.read_csv(file_path)
    df = preprocess(df)

    elasticity_df = compute_elasticities(df)
 
    df = df.merge(elasticity_df, on=["name"], how = 'inner')

    df["cost"] = df["Discount_price"] * 0.6
    
    price_slots_num = 10
    
    df["price_range"] = df["Actual_price"].apply(lambda x: np.linspace(0.5*x, 1.5*x, num=price_slots_num).tolist())

    products_num= len(df[["name"]].value_counts())

    env = DynamicPricingForECommerce.DynamicPricingForECommerceEnv(df, products_num, price_slots_num)

    agent = QLearningAgent.QLearningAgent(env)

    reward_history = agent.train_q_learning(env, episodes=50)
    episodes_list = list(range(len(reward_history)))
    plt.plot(episodes_list, reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Q learning on {}'.format("dynamic pricing"))
    plt.show()
    best_actions = [np.argmax(agent.Q[i]) for i in range(len(agent.Q))]
    df = df.iloc[:len(best_actions)]  # Ensure lengths match
    df["optimal_price"] = [df["price_range"].iloc[i][best_actions[i]] for i in range(len(best_actions))]
    print("\nLearned optimal price per product:")
    print(df[["name", "Actual_price", "Impression", "elasticity", "Week", "cost", "optimal_price"]].describe())
    
