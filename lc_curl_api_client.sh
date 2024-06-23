nohup curl --location --request POST 'http://localhost:8000/generate' --header 'Content-Type: application/json' --data-raw '{"prompt":"Report 2024 Discover Financial Balance Sheet. Use internet to find the information", "max_tokens" : 1000 }' &

nohup curl --location --request POST 'http://localhost:8000/generate' --header 'Content-Type: application/json' --data-raw '{"prompt":"Report First and Second Quarter of 2023 Discover Financial Balance Sheet and compare the balance sheets. Use internet to find the information", "max_tokens" : 1000 }' &
