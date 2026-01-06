# Build Stage
FROM golang:alpine AS builder

# Install git for fetching dependencies
RUN apk add --no-cache git

WORKDIR /app

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
# We build the 'main' package located in cmd/api/main.go
# Output binary named 'server'
RUN CGO_ENABLED=0 GOOS=linux go build -o server ./cmd/api

# Final Stage
FROM alpine:latest

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /app/server .

# Copy config file (if needed at runtime, usually good to have default)
COPY config.yaml .

# Expose the API port
EXPOSE 8081

# Run the server
CMD ["./server"]
