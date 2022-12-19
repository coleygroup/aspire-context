build:
	docker compose up -d

rebuild:
	docker compose up -d --build

down:
	docker compose down

stop:
	docker compose stop

debug:
	uvicorn app.main:app --host localhost --port 8000 --reload