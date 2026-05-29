.PHONY: dev dev-api dev-frontend build-frontend

dev:
	$(MAKE) -j2 dev-api dev-frontend

dev-api:
	uvicorn rembrandt.web.app:create_app --factory --host 127.0.0.1 --port 8000 --reload

dev-frontend:
	cd frontend && yarn dev

build-frontend:
	cd frontend && yarn install && yarn build
