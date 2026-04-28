package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
)

func main() {
	client := anthropic.NewClient()
	scanner := bufio.NewScanner(os.Stdin)
	getUserMsg := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}
	tools := []ToolDefinition{}
	agent := NewAgent(&client, getUserMsg, tools)
	err := agent.Launch(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *anthropic.Client, getUserMsg func() (string, bool), tools []ToolDefinition) *Agent {
	return &Agent{
		client:     client,
		getUserMsg: getUserMsg,
		tools:      tools,
	}
}

/* Let agent use tools
- user tells model what tools are available
- model decides if it wants to use a tool at some point and specifically send request for that tool
- user runs tool on its send and send back any outputs if available
*/

// A tool needs to be defined in a certain way
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(input json.RawMessage) (string, error)
}

// Tool: read_file
var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use it on files when needed, won't work with directories.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description: "the relative path of a file in the current working directory"`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		panic(err)
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}
func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)

	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}

type Agent struct {
	client     *anthropic.Client // anthropic client will look for ANTHROPIC_API_KEY
	getUserMsg func() (string, bool)
	tools      []ToolDefinition
}

func (a *Agent) Launch(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}
	fmt.Println("Hello! (exit with <C-c>")

	for {
		// user prompt head
		fmt.Print("User: ")
		userInput, ok := a.getUserMsg()
		if !ok {
			break
		}
		userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMsg)
		msg, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		for _, content := range msg.Content {
			switch content.Type {
			case "text":
				fmt.Printf("Claude: %s\n", content.Text)
			}
		}
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	// send available tools to model
	anthrophicTools := []anthropic.ToolUnionParam{}
	for _, tool := range a.tools {
		anthrophicTools = append(anthrophicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: tool.InputSchema,
			},
		})
	}
	msg, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_6,
		MaxTokens: int64(1024),
		Messages:  conversation,
	})
	return msg, err
}
