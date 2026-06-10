FROM nginx:1.27-alpine

COPY index.html /usr/share/nginx/html/index.html
COPY default.conf.template /etc/nginx/templates/default.conf.template

# Substituted into the nginx config at container start by the base image entrypoint.
ENV API_UPSTREAM=asr:8000

EXPOSE 8080
