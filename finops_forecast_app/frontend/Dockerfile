# Stage 1: Build the React app
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package.json ./
# If using yarn:
# COPY yarn.lock ./
# RUN yarn install --frozen-lockfile

# If using npm:
COPY package-lock.json ./
RUN npm ci

# Copy the rest of the app's source code
COPY . .

# Build the app
RUN npm run build
# If using yarn:
# RUN yarn build

# Stage 2: Serve the built app with Nginx
FROM nginx:1.25-alpine

# Copy the build output from the builder stage to Nginx's html directory
COPY --from=builder /app/build /usr/share/nginx/html

# (Optional) Copy a custom Nginx configuration if needed
# For a typical SPA, you might need to configure Nginx to redirect all
# non-file requests to index.html to support client-side routing.
# COPY nginx.conf /etc/nginx/conf.d/default.conf
# Example nginx.conf content for SPA:
# server {
#   listen 80;
#   server_name localhost;
#
#   root /usr/share/nginx/html;
#   index index.html index.htm;
#
#   location / {
#     try_files $uri $uri/ /index.html;
#   }
#
#   # Proxy API requests to the backend (if not handled by an external gateway/load balancer)
#   # This is an alternative to the React proxy in package.json if running frontend standalone
#   # location /api/v1 {
#   #   proxy_pass http://backend_service_name:8000; # Replace with your backend service name in Docker network
#   #   proxy_set_header Host $host;
#   #   proxy_set_header X-Real-IP $remote_addr;
#   #   proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#   #   proxy_set_header X-Forwarded-Proto $scheme;
#   # }
# }


# Expose port 80 for Nginx
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
