.PHONY: all build test install clean

BINARY_NAME=omm

all: build test

build:
	go build -o $(BINARY_NAME)

test:
	go test -v ./...

install:
	go install

clean:
	rm -f $(BINARY_NAME)
	go clean
