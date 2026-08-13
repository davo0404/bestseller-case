# BESTSELLER Data Scientist Interview Case


## Forecasting SKU-Level Demand

Forecasting is an essential part of managing the supply chain and to ensure our products are always in stock. In BESTSELLER we have a range of tools to help planners maintain stock levels. In one of these tools, we need to forecast at a weekly granularity four weeks ahead what the demand is at SKU-level.

### What is a SKU?

A SKU in BESTSELLER is a combination of a style, a colour and a size.

**Examples:**
- Round Neck Knitted Pullover, Black, L
- Knitted Pencil Skirt, Natural Melange, XS

---

## Questions

### 1. Model Design

**We want to build an ML model to solve this forecasting problem.**

What kind of model and predictor variables would you consider for this forecasting problem?


Given a specific style and colour, the above model will give forecasts for each size, and we can calculate the proportion each size has of a total. A priori, there is no mechanism to ensure these proportions are sensible, i.e., you get something like Figure 1 and not like Figure 2.

![alt text](image.png)


---

### 2. Size Split Consistency

**What can be done to ensure our SKU-forecasts does not in combination give a wrong size split?**

Stockouts represented in the historical sales data will be problematic for the model if not accounted for.

---

### 3. Handling Stockouts

**What options or strategies would you pursue to account for stockouts, so that we use SKU demand rather than SKU sales as target for our forecasting model?**

---

## Guidelines

The purpose of the exercise is to have a foundation for discussion. Please only use a couple of hours on this exercise – we are not looking for a fully detailed solution.

If there are any questions, please do reach out.