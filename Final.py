import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

def reg_anal(preg_df):
    # Remove NULLs
    preg_df = preg_df[~preg_df['Glucose'].isna()]
    preg_df = preg_df[~preg_df['BloodPressure'].isna()]
    preg_df = preg_df[~preg_df['BMI'].isna()]

    X = preg_df[["Glucose", "BloodPressure", "BMI"]]
    X = sm.add_constant(X)
    Y = preg_df["Outcome"]

    reg_model = sm.OLS(Y, X)
    reg_result = reg_model.fit()

    print(reg_result.summary())


def test(preg_df):
    # Remove NULLs
    preg_df = preg_df[~preg_df['Glucose'].isna()]
    preg_df = sm.add_constant(preg_df)

    X = preg_df[["const", "Glucose"]]
    Y = preg_df["Outcome"]

    logit_model = sm.Logit(Y, X)
    logit_result = logit_model.fit()

    print(logit_result.summary())


def scatter(preg_df):
    # Glucose v BloodPressure
    plt.scatter(preg_df["Glucose"], preg_df["BloodPressure"])
    plt.xlabel("Glucose")
    plt.ylabel("Blood Pressure")
    plt.title("Glucose vs Blood Pressure")
    plt.grid(True)
    plt.show()

    # Glucose v BMI
    plt.scatter(preg_df["Glucose"], preg_df["BMI"])
    plt.xlabel("Glucose")
    plt.ylabel("BMI")
    plt.title("Glucose vs BMI")
    plt.grid(True)
    plt.show()

    # Covariance
    cov = {
        "glu" : preg_df["Glucose"],
        "bp" : preg_df["BloodPressure"],
        "bmi" : preg_df["BMI"]
    }
    cov_df = pd.DataFrame(cov)
    cov_matrix = cov_df.cov()
    print(cov_matrix)


def dist(preg_df):
    # Remove NULLs
    preg_df = preg_df[~preg_df['Glucose'].isna()]

    # Creates a list of distributions
    dist_list = [stats.norm, stats.expon, stats.lognorm]

    best_fit = None
    best_param = None
    best_sse = np.inf

    # Find out which distribution is the best fit from our list
    for distribution in dist_list:
        param = distribution.fit(preg_df["Glucose"])
        sse = np.sum((distribution.pdf(preg_df["Glucose"], *param) - preg_df["Glucose"]) **2)

        if sse < best_sse:
            best_fit = distribution
            best_param = param
            best_sse = sse

    # Plot best distribution
    sns.histplot(preg_df["Glucose"], bins=10, kde=True, label='Data Histogram')
    x = np.linspace(preg_df["Glucose"].min(), preg_df["Glucose"].max(), 100)
    plt.plot(x, best_fit.pdf(x, *best_param), label=f'{best_fit.name} Distribution')
    plt.xlabel("Glucose")
    plt.ylabel("Frequency / Probability Density")
    plt.title("Best Fit Distribution of Blood Glucose")
    plt.legend()
    plt.grid(True)
    plt.show()


def cdf(preg_df):
    pregnancies_sorted = np.sort(preg_df["Pregnancies"])
    cdf = np.arange(1, len(pregnancies_sorted) + 1) / len(pregnancies_sorted)

    plt.plot(pregnancies_sorted, cdf, marker="o")
    plt.xticks(range(0, 18))
    plt.xlabel("Pregnancies")
    plt.ylabel("Probability")
    plt.title("CDF for Pregnancies")
    plt.grid(True)
    plt.show()


def pmf(preg_df):
    scenario1 = preg_df[preg_df["Age"] <= 30]
    scenario2 = preg_df[preg_df["Age"] >= 31]

    pmf_scenario1 = scenario1["BMI"].value_counts(normalize=True).sort_index()
    pmf_scenario2 = scenario2["BMI"].value_counts(normalize=True).sort_index()

    differences = (pmf_scenario2 - pmf_scenario1) * 100

    plt.figure()
    plt.bar(differences.index, differences.values)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("BMI")
    plt.ylabel("Relative Percent Difference")
    plt.title("Relative Difference in PMFs for BMI between Age Scenarios")
    plt.grid(True)
    plt.show()


def a_histogram(preg_df):
    # Histogram
    plt.hist(preg_df["Age"], bins=20)
    plt.xlabel("Age in Years")
    plt.ylabel("Frequency")
    plt.title("Age of Patient Histogram")
    plt.grid(True)
    plt.show()

    # Characteristics
    a_mean = preg_df["Age"].mean()
    a_mode = preg_df["Age"].mode()
    a_spread = preg_df["Age"].std()
    a_tail = preg_df["Age"].tail()

    print(f"Mean: {a_mean}")
    print(f"Mode: {a_mode}")
    print(f"spread {a_spread}")
    print(f"Tail: {a_tail}")


def bmi_histogram(preg_df):
    # Histogram
    plt.hist(preg_df["BMI"], bins=20)
    plt.xlabel("Body Mass Index")
    plt.ylabel("Frequency")
    plt.title("BMI Histogram")
    plt.grid(True)
    plt.show()

    # Characteristics
    bmi_mean = preg_df["BMI"].mean()
    bmi_mode = preg_df["BMI"].mode()
    bmi_spread = preg_df["BMI"].std()
    bmi_tail = preg_df["BMI"].tail()

    print(f"Mean: {bmi_mean}")
    print(f"Mode: {bmi_mode}")
    print(f"spread {bmi_spread}")
    print(f"Tail: {bmi_tail}")


def b_histogram(preg_df):
    # Histogram
    plt.hist(preg_df["BloodPressure"], bins=20)
    plt.xlabel("Diastolic Blood Pressure")
    plt.ylabel("Frequency")
    plt.title("Blood Pressure Histogram")
    plt.grid(True)
    plt.show()

    # Characteristics
    b_mean = preg_df["BloodPressure"].mean()
    b_mode = preg_df["BloodPressure"].mode()
    b_spread = preg_df["BloodPressure"].std()
    b_tail = preg_df["BloodPressure"].tail()

    print(f"Mean: {b_mean}")
    print(f"Mode: {b_mode}")
    print(f"spread {b_spread}")
    print(f"Tail: {b_tail}")


def g_histogram(preg_df):
    # Histogram
    plt.hist(preg_df["Glucose"], bins=20)
    plt.xlabel("Plasma Glucose Concentration")
    plt.ylabel("Frequency")
    plt.title("Glucose Histogram")
    plt.grid(True)
    plt.show()

    # Characteristics
    g_mean = preg_df["Glucose"].mean()
    g_mode = preg_df["Glucose"].mode()
    g_spread = preg_df["Glucose"].std()
    g_tail = preg_df["Glucose"].tail()

    print(f"Mean: {g_mean}")
    print(f"Mode: {g_mode}")
    print(f"spread {g_spread}")
    print(f"Tail: {g_tail}")


def p_histogram(preg_df):
    # Histogram
    plt.hist(preg_df["Pregnancies"], bins=range(0,18))
    plt.xticks(range(0,18))
    plt.xlabel("Number of Pregnancies")
    plt.ylabel("Frequency")
    plt.title("Pregnancies Histogram")
    plt.grid(True)
    plt.show()

    # Characteristics
    p_mean = preg_df["Pregnancies"].mean()
    p_mode = preg_df["Pregnancies"].mode()
    p_spread = preg_df["Pregnancies"].std()
    p_tail = preg_df["Pregnancies"].tail()

    print(f"Mean: {p_mean}")
    print(f"Mode: {p_mode}")
    print(f"spread {p_spread}")
    print(f"Tail: {p_tail}")


def create_df():
    preg_df = pd.read_csv(r'C:\Users\User\PycharmProjects\DSC530-FINAL\diabetes.csv', encoding='latin1')
    preg_df["BloodPressure"] = preg_df["BloodPressure"].replace(0, np.nan)
    preg_df["BMI"] = preg_df["BMI"].replace(0, np.nan)
    preg_df["Glucose"] = preg_df["Glucose"].replace(0, np.nan)
    return preg_df


def main():
    preg_df = create_df()

    # Histograms
    p_histogram(preg_df)
    g_histogram(preg_df)
    b_histogram(preg_df)
    bmi_histogram(preg_df)
    a_histogram(preg_df)

    # Testing
    pmf(preg_df)
    cdf(preg_df)
    dist(preg_df)
    scatter(preg_df)
    test(preg_df)
    reg_anal(preg_df)


if __name__ == '__main__':
    main()
