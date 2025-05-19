# SheetMinds API

This API provides a REST interface to the SheetMinds CSV/Excel Data Analysis system. It allows you to set a data source, ask questions about your data, and get profiling information.

## Getting Started

### Prerequisites

Ensure you have all the requirements installed:

```bash
pip install -r requirements.txt
```

### Running the API

Start the API server:

```bash
python api.py
```

The server will run on `http://localhost:5000` by default.

## API Endpoints

### Set CSV File Path

```
POST /api/set_csv_path
```

Set the path to the CSV file to analyze.

**Request Body:**

```json
{
  "path": "/path/to/your/file.csv"
}
```

**Response:**

```json
{
  "success": true,
  "message": "CSV path set to: /path/to/your/file.csv"
}
```

### Ask a Question

```
POST /api/ask
```

Ask a natural language question about the CSV data.

**Request Body:**

```json
{
  "question": "What is the average value in column X?",
  "path": "/path/to/your/file.csv" // Optional, overrides the global path
}
```

## Usage Examples

### Using curl

````bash
# Set the CSV path
curl -X POST http://localhost:5000/api/set_csv_path \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/your/file.csv"}'

# Ask a question
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average salary?"}'

### Using Python requests

```python
import requests

# Set the CSV path
response = requests.post(
    "http://localhost:5000/api/set_csv_path",
    json={"path": "/path/to/your/file.csv"}
)
print(response.json())

# Ask a question
response = requests.post(
    "http://localhost:5000/api/ask",
    json={"question": "How many entries have missing values?"}
)
print(response.json())
````

## Integrating with Frontend Applications

You can easily integrate this API with frontend frameworks like React, Vue, or Angular. Here's a simple JavaScript example:

```javascript
// Set the CSV path
fetch("http://localhost:5000/api/set_csv_path", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    path: "/path/to/your/file.csv",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));

// Ask a question
fetch("http://localhost:5000/api/ask", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    question: "What are the top 5 values in column Z?",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```
