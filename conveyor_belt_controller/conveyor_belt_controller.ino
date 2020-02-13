const char startChar = 'A';
const char stopChar = 'S';
const char startCommandChar = '(';
const char stopCommandChar = ')';

auto rele_pin = 8;
char command = 0;
bool reading_command = false;
bool can_execute = false;


void read_and_interpret_byte()
{ 
   if (Serial.available() > 0)
   {
      char byte_read = Serial.read();
      if (byte_read == startChar)
      {
         command = startChar;
      }
      else if (byte_read == stopChar)
      {
         command = stopChar;
      }
      else if (byte_read == startCommandChar)
      {
         reading_command = true;
      }
      else if (byte_read == stopCommandChar)
      {
         reading_command = false;
         can_execute = true;
      }
   }
}

void execute_command()
{
   if (can_execute)
   {
      if (command == startChar)
      {
         digitalWrite(rele_pin, LOW);
      }
      else if (command == stopChar)
      {
         digitalWrite(rele_pin, HIGH);
      }
      can_execute = false;
   }
}


void setup()
{
   Serial.begin(9600);
   pinMode(rele_pin, OUTPUT);
   digitalWrite(rele_pin, HIGH);
}

void loop()
{
   read_and_interpret_byte();
   execute_command();
}
