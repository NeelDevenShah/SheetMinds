# SheetMind API Documentation

SheetMind provides a RESTful API for analyzing structured data files (CSV, Excel) using AI-powered analysis. This document describes the available endpoints and how to use them.

## Base URL

All API endpoints are relative to the base URL where the API is hosted (e.g., `http://localhost:5000`).

## Authentication

Currently, the API does not require authentication. However, in a production environment, you should implement proper authentication.

## Rate Limiting

By default, the API does not enforce rate limiting. In production, consider adding rate limiting to prevent abuse.

## Endpoints

### 1. Upload File

Upload a file for analysis.

- **URL**: `/api/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required): The file to upload (CSV, XLSX, XLS, or Parquet)
- **Response** (success):
  ```json
  {
    "success": true,
    "data": {
      "filename": "unique-filename.csv",
      "filepath": "/path/to/uploads/unique-filename.csv",
      "size": 1024,
      "content_type": "text/csv"
    },
    "timestamp": "2023-01-01T12:00:00.000000"
  }
  ```
- **Response** (error):
  ```json
  {
    "success": false,
    "error": "No file part in the request",
    "timestamp": "2023-01-01T12:00:00.000000"
  }
  ```

### 2. Analyze Data

Analyze data using a natural language query.

- **URL**: `/api/analyze`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "query": "What is the average price by category?",
    "file_path": "/path/to/your/file.csv"
  }
  ```
- **Response** (success):
  ```json
  {
    "success": true,
    "data": {
      "result": "The average price by category is...",
      "explanation": "I calculated the average price for each category by...",
      "code": "df.groupby('category')['price'].mean()",
      "data_preview": {
        "columns": ["category", "average_price"],
        "data": [
          ["Electronics", 599.99],
          ["Clothing", 49.99],
          ["Books", 19.99]
        ]
      }
    },
    "timestamp": "2023-01-01T12:00:00.000000"
  }
  ```
- **Response** (error):
  ```json
  {
    "success": false,
    "error": "File not found: /path/to/your/file.csv",
    "timestamp": "2023-01-01T12:00:00.000000"
  }
  ```

## Error Handling

All error responses follow this format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "timestamp": "2023-01-01T12:00:00.000000"
}
```

Common HTTP status codes:

- `200 OK`: The request was successful
- `400 Bad Request`: The request was invalid (e.g., missing parameters)
- `404 Not Found`: The requested resource was not found
- `500 Internal Server Error`: An error occurred on the server

## Example Usage

### Using cURL

1. **Upload a file**:

   ```bash
   curl -X POST -F "file=@/path/to/your/data.csv" http://localhost:5000/api/upload
   ```

2. **Analyze data**:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
     "query": "What is the average price by category?",
     "file_path": "/path/to/your/data.csv"
   }' http://localhost:5000/api/analyze
   ```

### Using Python

```python
import requests

# Upload a file
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/upload',
        files={'file': f}
    )
file_info = response.json()

# Analyze data
response = requests.post(
    'http://localhost:5000/api/analyze',
    json={
        'query': 'What is the average price by category?',
        'file_path': file_info['data']['filepath']
    }
)
analysis = response.json()


## Rate Limiting and Best Practices

- **Batch Requests**: When possible, combine multiple operations into a single request.
- **Caching**: Cache responses when appropriate to reduce server load.
- **Error Handling**: Always check the response status code and handle errors gracefully.
- **Pagination**: For large datasets, implement pagination in your client application.

## Security Considerations

- **File Uploads**: Always validate file types and scan for malicious content.
- **Sensitive Data**: Avoid sending sensitive information in requests or responses.
- **HTTPS**: Always use HTTPS in production to encrypt data in transit.
- **Authentication**: Implement proper authentication and authorization in production environments.

## Support

For support, please contact [neeldevenshah@gmail.com](mailto:neeldevenshah@gmail.com) or open an issue on our [GitHub repository](https://github.com/NeelDevenshah/SheetMinds).
```
