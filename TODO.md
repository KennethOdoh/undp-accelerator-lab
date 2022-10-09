1. Request for the missing lat and long from the data providers since they can figure it out from the original submission

2. The coulmn 'Unit Cost in USD' consists of a lot of currencies. we need to extract the values and convert to USD as standard. In addition, we need to remove the texts in the column. Also replace ', 'Undefined, and similar values with 'NaN' since this is a number column. Where values are given as a range, we can take average.

3. We can use the auto-detect and translate feature of Google translate to translate all texts TO English.