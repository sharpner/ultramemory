package store

import "fmt"

const (
	DBFormatKey     = "db_format"
	DBFormatOllama  = "ollama-v1"
	DBFormatMistral = "mistral-v1"
)

type DBFormatMismatchError struct {
	Expected string
	Actual   string
}

func (e *DBFormatMismatchError) Error() string {
	return fmt.Sprintf("wrong db format: expected %s, found %s", e.Expected, e.Actual)
}

func CurrentDBFormat() string {
	return currentDBFormat
}
