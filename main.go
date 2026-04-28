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
	tools := []ToolDefinition{ReadFileDefinition}
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
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
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

	// when a model request tool use, toggle this off. Basically, when detecting agent requires tool use, dont do user input but execute the tool, grab tool output (if any) and send message back to tool.
	readUserInput := true
	for {
		if readUserInput {
			// user prompt head
			fmt.Print("User: ")
			userInput, ok := a.getUserMsg()
			if !ok {
				break
			}
			userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
			conversation = append(conversation, userMsg)
		}
		msg, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, msg.ToParam())

		toolResults := []anthropic.ContentBlockParamUnion{}
		for _, content := range msg.Content {
			switch content.Type {
			case "text":
				fmt.Printf("Claude: %s\n", content.Text)
			case "tool_use":
				result := a.executeTool(content.ID, content.Name, content.Input)
				toolResults = append(toolResults, result)
			}
		}
		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}
		// when tool has results, need to send back to model to process
		readUserInput = false
		conversation = append(conversation, anthropic.NewUserMessage(toolResults...))
	}

	return nil
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) anthropic.ContentBlockParamUnion {
	var toolDef ToolDefinition
	var found bool
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}
	if !found {
		return anthropic.NewToolResultBlock(id, "tool not found", true)
	}
	fmt.Printf("tool %s(%s)\n", name, input)
	response, err := toolDef.Function(input)
	if err != nil {
		return anthropic.NewToolResultBlock(id, err.Error(), true)
	}
	return anthropic.NewToolResultBlock(id, response, false)
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
		Model:     anthropic.ModelClaudeHaiku4_5,
		MaxTokens: int64(1024),
		Messages:  conversation,
		Tools:     anthrophicTools,
	})
	return msg, err
}
